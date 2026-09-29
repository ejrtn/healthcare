import os

# 실측 확인: sentence-transformers/CrossEncoder가 이미 로컬에 캐시된 모델을
# 불러올 때도, "캐시가 최신인지" 확인하려고 Hugging Face Hub에 HEAD/GET 요청을
# 수십 번 보낸다 — MCP 서버를 통해 재현했을 때 이 네트워크 확인 과정만으로
# 임베딩+리랭커 로딩에 40초 가까이 걸렸다(전체 첫 호출 104초 중 대부분).
# 모델을 업데이트할 계획이 없으므로, 네트워크 확인 자체를 꺼서 로컬 캐시를
# 그대로 쓰게 한다 — 반드시 sentence_transformers를 import하기 전에 설정해야
# 한다.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import json
import asyncio
import threading
from typing import List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
import chromadb

# Windows 콘솔 기본 인코딩(cp949)에서 이모지 등 유니코드 print()가 죽는 문제 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

app = FastAPI()

# static 폴더 연결 (HTML 파일용)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 1. 임베딩 모델 설정
# finetune/ko-sroberta-finetuned-v2(TripletLoss 파인튜닝 버전)를 실제 서빙에
# 붙여서 실측해봤는데, 카테고리 라우팅(리랭킹 후)은 93.9%→97.0%로 개선됐지만
# 거절 정확도(55.6%→44.4%)와 증상 매칭(56.5%→47.8%)이 둘 다 악화됐다 —
# 학습 데이터가 25~30건뿐이라 33문항 카테고리 질문에는 과최적화되고 전체적인
# 임베딩 공간의 범용성은 오히려 깎인 것으로 보인다(finetune/RESULTS.md 참고).
# 그래서 원본으로 되돌린다.
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 2. ChromaDB 영구 저장 클라이언트 (vector_db 폴더에 디스크 저장)
chroma_client = chromadb.PersistentClient(path="vector_db")
collection = chroma_client.get_or_create_collection(
    name="medical_knowledge_v2",  # 청킹 스키마 변경으로 컬렉션명 갱신 (기존 컬렉션과 분리)
    metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
)

# 2-1. LLM (Ollama 대신 llama-cpp-python으로 gguf 파일을 직접 로드)
# n_threads를 지정하지 않으면 llama-cpp-python이 보수적인 기본값을 쓰는 경우가 있어
# 물리 코어 수에 맞춰 명시적으로 지정한다. 예전엔 로컬 PC 코어 수(14)로 고정해뒀는데,
# Hugging Face Spaces 무료 CPU 티어(보통 vCPU 2개)에 그대로 가져가면 실제 코어보다
# 훨씬 많은 스레드가 서로 경쟁하면서 오히려 응답이 크게 느려지는 문제가 실측으로
# 확인됐다 — os.cpu_count()로 실행 환경의 실제 코어 수를 읽어 자동으로 맞춘다.
CPU_THREADS = os.cpu_count() or 4

# 8B 모델(4.6GB) 로딩 자체가 실측 100초 이상 걸린다(MCP 서버 타임아웃의 실제
# 원인 — embed_model/reranker는 HF_HUB_OFFLINE 적용 후 2초면 끝나는데, llm은
# 디스크에서 큰 파일을 읽는 시간이라 그 최적화로는 못 줄인다). search_medical_
# knowledge처럼 LLM을 아예 안 쓰는 도구까지 이 로딩을 기다리게 만들지 않도록,
# 다른 모델과 달리 이것만 지연 로딩한다 — 실제로 llm.create_chat_completion을
# 호출하는 시점에만 로드된다.
_llm = None
_llm_lock = threading.Lock()


def get_llm():
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = Llama(
                model_path="models/meta-llama-3.1-8b-instruct-q4_k_m.gguf",
                n_ctx=4096,
                n_threads=CPU_THREADS,        # 응답 생성(decode) 병렬도
                n_threads_batch=CPU_THREADS,  # 프롬프트 처리(prefill) 병렬도
                n_gpu_layers=0,  # GPU(CUDA build)로 가속하려면 늘리세요 (예: 20~35)
                chat_format="llama-3",
                verbose=False,
            )
    return _llm

# 2-2. 리랭커: 1차 벡터 검색 후보를 cross-encoder로 재정렬해 정확도를 높인다.
reranker = CrossEncoder("Dongjin-kr/ko-reranker", max_length=512)
RERANK_CANDIDATES = 8  # 벡터 검색으로 넉넉히 뽑아서 이 중 top_k만 리랭킹으로 추린다

# 3. 질문에서 병명을 감지하기 위한 병명 목록 (긴 이름부터 매칭해 부분 문자열 오매칭 방지)
DISEASES_PATH = "data/diseases.json"
KNOWN_DISEASES = []
if os.path.exists(DISEASES_PATH):
    with open(DISEASES_PATH, "r", encoding="utf-8") as f:
        KNOWN_DISEASES = json.load(f)

# 4. 병명별 실제 존재하는 카테고리 목록 (질문 의도 -> 카테고리 필터링에 사용)
DISEASE_CATEGORIES_PATH = "data/disease_categories.json"
DISEASE_CATEGORIES = {}
if os.path.exists(DISEASE_CATEGORIES_PATH):
    with open(DISEASE_CATEGORIES_PATH, "r", encoding="utf-8") as f:
        DISEASE_CATEGORIES = json.load(f)


def detect_diseases(text: str) -> List[str]:
    """
    질문 문자열에 등록된 병명이 몇 개든 포함되어 있으면 전부 반환.
    긴 병명부터 먼저 찾고, 매칭된 부분은 텍스트에서 지워서 그 안에 포함된
    짧은 병명(예: "간염"이 "만성 간염" 안에 들어있는 경우)이 중복으로 잡히지 않게 한다.
    결과는 질문에 등장한 순서대로 정렬해서, 답변도 사용자가 물어본 순서를 따르게 한다.
    """
    found = []
    remaining = text
    for disease in KNOWN_DISEASES:
        if disease in remaining:
            found.append(disease)
            remaining = remaining.replace(disease, " ", 1)
    return sorted(found, key=lambda d: text.find(d))


def guess_standard_term(user_text: str) -> str:
    """
    규칙 기반 병명 매칭과 임베딩 검색이 둘 다 실패했을 때("디스크"처럼 등록된
    정식 명칭의 일부만 말한 경우 등), LLM에게 이 표현이 어떤 질환/증상의
    구어체·줄임말인지 짧게 추측하게 한다.

    예전에 시도했던 "LLM에게 tool calling으로 병명/카테고리를 직접 분류시키는"
    에이전틱 실험은 정확도 0~43.8%로 규칙 기반(75~87%)보다 크게 떨어졌다
    (finetune/RESULTS.md 참고) — 그건 "여러 필드를 구조화해서 정확한 포맷으로
    뽑아내는" 작업이라 로컬 8B 모델이 약한 지점이었다. 여기서는 그것과 성격이
    다르다: "이 단어가 뭘 뜻하는지" 한 단어로 추측하는 단순 지식 회상이라,
    실패해도 그냥 원래 폴백 메시지로 넘어가면 되므로 부담 없이 시도해본다.

    출력이 비정상적으로 길거나("모름"이 아닌 애매한 답), 개행이 섞여 있으면
    신뢰할 수 없다고 보고 빈 문자열을 반환해 호출부가 원래 폴백으로 넘어가게 한다.
    """
    try:
        response = get_llm().create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "사용자가 입력한 짧은 표현이 어떤 질환이나 증상의 구어체·줄임말인지 "
                        "판단해서, 정식 의학 명칭 하나만 답하세요. 확실하지 않으면 정확히 "
                        "'모름'이라고만 답하세요. 설명이나 다른 말은 절대 추가하지 마세요."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            max_tokens=32,
        )
        guess = response["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

    if not guess or guess == "모름" or len(guess) > 20 or "\n" in guess:
        return ""
    return guess


# 질문에 특정 주제 단어가 있으면 그 주제와 관련된 카테고리로 검색 범위를 좁힌다.
# "치료"처럼 흔한 단어는 개요성 질문("치료법이 뭐야?")에도 섞여 들어오므로,
# 특정 주제 단어를 먼저 확인하고, 아무 주제도 안 걸리면 그때만 개요/정의로 본다.
CATEGORY_INTENTS = {
    "원인": {"triggers": ["원인", "왜 생기", "왜 걸리", "이유"], "keywords": ["원인", "역학", "병태생리"]},
    "증상": {"triggers": ["증상", "증세"], "keywords": ["증상", "종류"]},
    "진단": {"triggers": ["진단", "검사"], "keywords": ["진단", "검사", "평가"]},
    "약물": {"triggers": ["약", "복용", "처방"], "keywords": ["약물"]},
    "치료": {"triggers": ["치료", "낫는", "고치"], "keywords": ["치료"]},
    "예방": {"triggers": ["예방", "조심", "주의"], "keywords": ["예방", "위험요인"]},
    "합병증": {"triggers": ["합병증", "부작용"], "keywords": ["합병증"]},
    "관리": {"triggers": ["관리", "생활", "일상", "평소"], "keywords": ["관리", "실천", "생활습관"]},
}
OVERVIEW_INTENT = {"triggers": ["뭐야", "무엇", "이란", "란?", "정의", "설명"], "keywords": ["개요", "정의", "요약"]}


def detect_category_keywords(text: str) -> List[str]:
    """질문에서 의도를 감지해 카테고리 이름 매칭에 쓸 키워드 목록을 반환."""
    keywords = []
    for intent in CATEGORY_INTENTS.values():
        if any(trigger in text for trigger in intent["triggers"]):
            keywords.extend(intent["keywords"])
    if keywords:
        return keywords
    if any(trigger in text for trigger in OVERVIEW_INTENT["triggers"]):
        return OVERVIEW_INTENT["keywords"]
    return []


def resolve_categories(disease: str, category_keywords: List[str]) -> List[str]:
    """해당 병명이 실제로 갖고 있는 카테고리 중 키워드를 포함하는 것만 골라낸다."""
    if not category_keywords:
        return []
    actual_categories = DISEASE_CATEGORIES.get(disease, [])
    return [c for c in actual_categories if any(k in c for k in category_keywords)]

@app.on_event("startup")
def load_data():
    """
    서버 시작 시 ChromaDB에 데이터가 없을 때만 임베딩 후 저장.
    이미 데이터가 있으면 기존 DB를 그대로 사용 (빠른 재시작).
    """
    existing_count = collection.count()

    if existing_count > 0:
        print(f"✅ ChromaDB에 기존 데이터 {existing_count}건 로드 완료 (임베딩 생략)")
        return

    data_path = "data/medical_knowledge.jsonl"
    if not os.path.exists(data_path):
        print("⚠️  medical_knowledge.jsonl 파일이 없습니다. data/parse_xml.py를 먼저 실행하세요.")
        return

    print("📂 의료 데이터 로딩 중...")
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("⚠️  데이터가 비어있습니다.")
        return

    # 배치 처리 (메모리 절약: 500건씩)
    BATCH_SIZE = 500
    total = len(records)
    print(f"🔄 총 {total}건 임베딩 시작...")

    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_docs = [r["content"] for r in batch]
        # 임베딩할 때만 병명/카테고리를 붙여서, "병명이 뭐야?" 같은 질문도
        # 본문에 병명이 직접 언급되지 않은 청크(예: 개요)와 잘 매칭되도록 한다.
        # 저장/표시용 documents는 순수 content로 유지한다.
        batch_embed_texts = [f"{r['disease']} {r['category']}: {r['content']}" for r in batch]
        batch_embeddings = embed_model.encode(batch_embed_texts).tolist()
        batch_ids = [f"doc_{i + j}" for j in range(len(batch))]
        batch_meta = [
            {
                "disease": r["disease"],
                "category": r["category"],
                "chunk_index": r["chunk_index"],
                "total_chunks": r["total_chunks"],
            }
            for r in batch
        ]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"  → {min(i + BATCH_SIZE, total)}/{total} 처리 완료")

    print(f"✅ 의료 데이터 {total}건 ChromaDB 저장 완료!")


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []  # [{"role": "user"|"assistant", "content": str}, ...] 프론트엔드가 매번 통째로 보냄


# 코사인 거리 0=완전일치, 2=완전반대.
# 0.7은 실측(eval/RESULTS.md 참고) 결과 완전히 무관한 질문(날씨/주식 등)도
# 0.54~0.61 사이로 걸러내지 못해서 0.5로 낮췄다. 관련 있는 질문 중 극히 일부
# (거리 0.55~0.6대의 애매한 케이스)가 덩달아 거절될 수 있지만, 의료 챗봇에서는
# 무관한 질문에 답을 지어내는 것보다 애매한 경우 거절하는 쪽이 안전하다.
THRESHOLD = 0.5


def build_context(docs, metadatas) -> str:
    parts = []
    for doc, meta in zip(docs, metadatas):
        disease = meta.get("disease", "")
        category = meta.get("category", "")
        parts.append(f"[{disease} - {category}]\n{doc}")
    return "\n\n".join(parts)


def collect_sources(metadatas, sources: List[dict]) -> None:
    """답변에 실제로 사용된 (병명, 카테고리) 출처를 중복 없이 sources에 추가한다."""
    for meta in metadatas:
        entry = {"disease": meta.get("disease", ""), "category": meta.get("category", "")}
        if entry not in sources:
            sources.append(entry)


def rerank(query_text: str, docs, distances, metadatas, top_k: int):
    """
    "관련 없음" 판단(THRESHOLD)을 통과한 후보들 중에서, cross-encoder로 top_k를 골라낸다.
    거리값(distances)은 재정렬 전 코사인 거리를 그대로 들고 다닌다 (출처 표시 등에서
    참고용으로만 쓰고, 최종 관련성 판단에는 쓰지 않는다).

    1등만 THRESHOLD를 넘는지 보는 건 "상대 평가"라 후보가 부족한 경우 8개 중
    뒤쪽 몇 개는 실제로는 THRESHOLD를 넘는(관련 없는) 것들로 채워질 수 있다.
    그런 약한 후보까지 리랭커에게 넘기면, 리랭커가 엉뚱하게 그걸 승격시킬 수 있으므로
    리랭킹 전에 후보군 자체를 절대 기준(THRESHOLD)으로 한 번 걸러낸다.
    """
    candidates = [
        (doc, dist, meta)
        for doc, dist, meta in zip(docs, distances, metadatas)
        if dist <= THRESHOLD
    ]
    docs = [c[0] for c in candidates]
    distances = [c[1] for c in candidates]
    metadatas = [c[2] for c in candidates]

    if len(docs) <= top_k:
        return docs, distances, metadatas

    pairs = [(query_text, doc) for doc in docs]
    scores = reranker.predict(pairs)
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
    return (
        [docs[i] for i in order],
        [distances[i] for i in order],
        [metadatas[i] for i in order],
    )


def vector_search(query_embedding, n_results: int):
    """병명이 감지되지 않았을 때(증상 기반 질문 등) 전체 컬렉션에서 후보를 넉넉히 뽑는다."""
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=max(n_results, RERANK_CANDIDATES),
        include=["documents", "distances", "metadatas"],
    )
    return results["documents"][0], results["distances"][0], results["metadatas"][0]


def search_disease(query_embedding, disease: str, category_keywords: List[str], n_results: int):
    """
    특정 병명으로 검색 범위를 좁혀서 후보를 가져온다.
    질문 의도(category_keywords)에 해당하는 카테고리가 그 병에 실제로 있으면
    그 카테고리로 한 번 더 좁히고, 없으면 병명 전체 범위에서 검색한다.
    """
    where = {"disease": disease}
    resolved_categories = resolve_categories(disease, category_keywords)
    if resolved_categories:
        where = {"$and": [{"disease": disease}, {"category": {"$in": resolved_categories}}]}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=max(n_results, RERANK_CANDIDATES),
        where=where,
        include=["documents", "distances", "metadatas"],
    )
    return results["documents"][0], results["distances"][0], results["metadatas"][0]


# 병명 없이 증상만 설명하는 질문에서, 투표로 모인 후보 질환이 많으면 하나로
# 단정하지 않고 되물어서 좁혀나간다 (vote_candidates_by_symptom 참고).
AMBIGUITY_MIN_DISEASES = 3    # 서로 다른 병이 이 개수 이상이면 애매하다고 판단
AMBIGUITY_MAX_DISEASES = 10   # 이 개수를 넘으면 목록을 보여주지 않고 증상을 더 물어본다
MAX_FOLLOWUP_ROUNDS = 3       # 후보가 안 좁혀져도 무한정 되묻지 않도록 하는 안전장치(주 기준은 후보 개수)


def build_symptom_query(message: str, history: List[dict]) -> str:
    """지금까지 대화에서 사용자가 말한 증상 설명을 이어붙여 누적 질의문을 만든다."""
    parts = [h.get("content", "") for h in history if h.get("role") == "user"]
    parts.append(message)
    return " ".join(p for p in parts if p)


def summarize_symptoms(message: str, history: List[dict]) -> str:
    """
    build_symptom_query()의 원문 이어붙이기를 LLM으로 한 번 더 정리해서 검색어로 쓴다.

    실측으로 확인된 문제: 대화가 길어질수록(예: 증상 6개를 한 메시지에 몰아넣은 경우)
    문장 하나에 여러 증상이 뒤섞여 임베딩이 흐려지고, "목이 가려워"에 반응한
    "두드러기"(피부), "가슴이 답답해"에 반응한 "위식도역류질환"(소화기)처럼
    임상적으로 무관한 후보가 섞여 들어왔다. 또 "약이 있나?" 같은 질문 형태에
    실려있는 카테고리 의도(치료/약물)가 그냥 버려지는 문제도 있었다.

    그래서 "그냥 요약해줘"가 아니라, 무엇을 남기고/빼고/우선할지 규칙과 few-shot
    예시로 구체적으로 명시한다(알약 프로젝트에서 확인된 것과 같은 원리 — 추상적
    지시보다 구체적 판별 기준이 로컬 8B 모델에는 더 잘 먹힌다).

    로컬 8B의 지시 이행이 항상 안정적인 건 아니라는 게 이 프로젝트에서 여러 번
    실측으로 확인됐으므로(RAGAS judge 파싱 성공률 40.6% 등), 출력이 이상하면
    build_symptom_query()의 원문 이어붙이기로 그대로 폴백한다.
    """
    raw = build_symptom_query(message, history)

    FEW_SHOT = [
        {"role": "user", "content": "기침이 심해지고 있는데 약이 있나?"},
        {"role": "assistant", "content": "기침 심함, 약물 치료"},
        {"role": "user", "content": "열이 나요 28도요 38도인가"},
        {"role": "assistant", "content": "발열 38도"},
        {
            "role": "user",
            "content": "기침이 나고 열도 나서 가슴이 답답해 호흡이 가빠지는 느낌이야 목도 따끔거리는데 목이 가려워",
        },
        {"role": "assistant", "content": "기침, 발열, 가슴 답답함, 호흡 곤란, 목 따끔거림, 목 가려움"},
    ]

    try:
        response = get_llm().create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "사용자가 여러 번에 걸쳐 말한 증상 설명을 검색에 쓸 짧은 문구로 정리하세요. "
                        "규칙: "
                        "1) '~있나요', '~인가요' 같은 질문 형태는 빼고 증상/상태만 남기되, "
                        "'약', '치료', '원인'처럼 무엇을 알고 싶어하는지 나타내는 단어는 그대로 남기세요. "
                        "2) 같은 증상을 여러 번 말했으면 한 번만 남기세요. "
                        "3) 체온처럼 값이 정정된 경우(예: 28도라고 했다가 38도라고 다시 말함) 최신 값만 남기세요. "
                        "4) 언급되지 않은 증상은 절대 지어내지 마세요. "
                        "5) 쉼표로 구분된 짧은 명사구 나열로만 답하고, 다른 설명은 절대 추가하지 마세요."
                    ),
                },
                *FEW_SHOT,
                {"role": "user", "content": raw},
            ],
            max_tokens=128,
        )
        summary = response["choices"][0]["message"]["content"].strip().strip('"').strip("'")
    except Exception as e:
        print(f"  [summarize_symptoms 오류] {e} -> 원문 이어붙이기로 폴백")
        return raw

    if not summary or "\n" in summary or len(summary) > 200:
        print(f"  [summarize_symptoms 이상 출력] {summary!r} -> 원문 이어붙이기로 폴백")
        return raw
    return summary


def vote_candidates_by_symptom(symptoms: List[str], n_results_per_symptom: int = 15):
    """
    여러 계통 증상이 한 벡터에 뭉쳐 흐려지는 문제(실측 확인: 기침+발열+가슴답답함+
    호흡곤란+목따끔거림+목가려움을 한 벡터로 검색하면 "두드러기", "위식도역류질환"
    처럼 무관한 후보가 섞여 들어옴)를 피하기 위해, summarize_symptoms()가 정리한
    쉼표 구분 증상 목록을 증상별로 따로따로 검색해서, 여러 검색에서 공통으로
    등장하는 병명에 표를 준다 — LLM의 판단(선택) 없이 순수 집계로 후보를 좁힌다.

    LLM에게 "후보 중 3개만 골라라"고 시켜본 적이 있는데(`finetune/RESULTS.md`
    "후보 추리기를 LLM 추론에 맡겨보기" 참고), "고르기"라는 지시를 무시하고 증상×
    후보를 전부 기계적으로 나열하다가 토큰 한도에 걸려 잘렸다 — 로컬 8B는 "여러
    선택지 중 일부만 고르는 판단"에 불안정하다는 게 재확인된 사례다. 이 함수는
    그 "고르기"를 LLM이 아니라 투표(집계)로 대체한다 — 검색·리랭킹처럼 "판단이
    아니라 계산으로 되는 일"은 전용 로직에 맡긴다는, 이 프로젝트가 원래 지켜온
    원칙의 연장선이다.

    반환: (ranked, overall_min_distance, pool)
    - ranked: 투표수 내림차순·동률이면 최소거리 오름차순으로 정렬된 후보 리스트,
      각 항목은 {"count", "distance", "doc", "meta"} (병명별 가장 가까운 청크 1개)
    - overall_min_distance: 전체 검색 중 최소 거리 — THRESHOLD 거절 판단에 사용
    - pool: 모든 증상 검색 결과를 합친 (doc, dist, meta) 튜플 리스트 — 후보가
      1~2개로 좁혀졌을 때 같은 병의 여러 청크를 모아 답변 컨텍스트를 풍부하게
      만드는 데 사용
    """
    votes: dict = {}
    overall_min_distance = None
    pool: list = []

    for symptom in symptoms:
        embedding = embed_model.encode([symptom]).tolist()
        docs, distances, metadatas = vector_search(embedding, n_results=n_results_per_symptom)
        if not docs:
            continue
        if overall_min_distance is None or distances[0] < overall_min_distance:
            overall_min_distance = distances[0]

        seen_this_symptom = set()
        for doc, dist, meta in zip(docs, distances, metadatas):
            if dist > THRESHOLD:
                continue
            pool.append((doc, dist, meta))
            disease = meta.get("disease")
            if disease in seen_this_symptom:
                continue  # 한 증상 검색에서 같은 병은 한 표만
            seen_this_symptom.add(disease)
            entry = votes.setdefault(disease, {"count": 0, "distance": dist, "doc": doc, "meta": meta})
            entry["count"] += 1
            if dist < entry["distance"]:
                entry["distance"] = dist
                entry["doc"] = doc
                entry["meta"] = meta

    ranked = sorted(votes.values(), key=lambda v: (-v["count"], v["distance"]))

    # 실측 확인: 최소 득표 기준 없이 "1표라도 받으면 후보"로 치면, 증상 하나하나의
    # 검색 결과를 합집합으로 모으는 셈이라 오히려 후보가 더 늘어난다(7개 증상 →
    # 32개 후보, 단일 벡터 검색의 19개보다 나쁨). "여러 증상에서 공통으로 나온
    # 병"만 진짜 후보로 남기기 위해, 증상이 2개 이상이면 최소 2표 이상만 남긴다.
    if len(symptoms) >= 2:
        filtered = [v for v in ranked if v["count"] >= 2]
        if filtered:
            ranked = filtered

    return ranked, overall_min_distance, pool


@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    query_embedding = embed_model.encode([user_input]).tolist()
    sources: List[dict] = []

    # 1. 질문에서 병명(들)과 카테고리 의도를 먼저 분류한다.
    diseases = detect_diseases(user_input)
    category_keywords = detect_category_keywords(user_input)

    print(f"질문: {user_input}")
    print(f"감지된 병명: {diseases}")
    print(f"감지된 카테고리 키워드: {category_keywords}")

    if not diseases:
        # 병명이 감지되지 않은 경우(증상 기반 질문 등): 전체 검색으로 폴백.
        # 지금까지 대화에서 나온 증상 설명을 LLM으로 정리해서 검색한다 (멀티턴 누적).
        # (LLM 호출이라 CPU 추론 기준 시간이 걸릴 수 있어 스레드에서 실행)
        symptom_query = await asyncio.to_thread(summarize_symptoms, user_input, request.history)
        print(f"  누적 증상 질의(LLM 정리): {symptom_query!r} (history {len(request.history)}턴)")

        # 증상을 하나로 뭉쳐서 검색하지 않고, 쉼표로 나눠 증상별로 따로 검색한 뒤
        # 투표로 후보를 모은다 (vote_candidates_by_symptom 참고 — 여러 계통 증상이
        # 섞일 때 한 벡터로 뭉치면 흐려지는 문제를 피하기 위한 설계).
        symptoms_list = [s.strip() for s in symptom_query.split(",") if s.strip()] or [symptom_query]
        ranked, overall_min_distance, pool = await asyncio.to_thread(
            vote_candidates_by_symptom, symptoms_list
        )

        if not ranked or overall_min_distance is None or overall_min_distance > THRESHOLD:
            # 규칙 기반 병명 매칭도, 임베딩 검색도 둘 다 실패한 경우 — "디스크"처럼
            # 등록된 정식 명칭("추간판탈출증(디스크)")의 일부만 말한 구어체/줄임말일
            # 가능성이 있으니, 완전히 포기하기 전에 LLM에게 한 번 물어봐서 맞으면
            # 되물어보고, 그마저도 안 되면 원래 폴백 메시지를 그대로 낸다.
            guess = await asyncio.to_thread(guess_standard_term, user_input)
            if guess:
                print(f"  근거 없음 -> LLM 추측: '{guess}' -> 되묻기")
                return {
                    "answer": f"혹시 '{guess}'을(를) 말씀하시는 건가요? 맞다면 정확한 병명으로 다시 질문해 주시면 자세히 안내해드릴게요.",
                    "sources": [], "is_followup": True,
                }
            return {"answer": "죄송합니다. 입력하신 증상에 대해 참고할 수 있는 의학적 근거가 데이터베이스에 없습니다. 정확한 진단을 위해 가까운 병원 방문을 권장합니다.", "sources": [], "is_followup": False}

        # 후보 질환이 여러 개로 갈리고, 아직 되물은 횟수가 한도 안이면 바로 답하지 않는다.
        # - 10개 넘게 갈리면 목록을 보여줘도 의미가 없으니 증상을 더 구체적으로 물어본다.
        # - 3~10개면 후보 목록을 그대로 보여주고, 증상을 더 주면 좁혀진다고 안내한다.
        assistant_turns = sum(1 for h in request.history if h.get("role") == "assistant")
        candidate_names = [v["meta"].get("disease") for v in ranked]

        if assistant_turns < MAX_FOLLOWUP_ROUNDS and len(candidate_names) > AMBIGUITY_MAX_DISEASES:
            print(f"  후보 {len(candidate_names)}개(10개 초과) -> 증상 추가 요청: {candidate_names}")
            return {
                "answer": (
                    f"입력하신 증상만으로는 해당할 수 있는 질병이 너무 많습니다 "
                    f"({len(candidate_names)}개 이상). 증상을 조금 더 구체적으로 알려주시면 "
                    "좁혀드릴게요 (예: 언제부터 시작됐는지, 다른 증상이 같이 있는지 등)."
                ),
                "sources": [], "is_followup": True,
            }

        if assistant_turns < MAX_FOLLOWUP_ROUNDS and len(candidate_names) >= AMBIGUITY_MIN_DISEASES:
            print(f"  후보 {len(candidate_names)}개 -> 목록 제시: {candidate_names}")
            listing = "\n".join(f"- {name}" for name in candidate_names)
            return {
                "answer": (
                    f"입력하신 증상으로는 다음 질병들이 예상됩니다:\n{listing}\n\n"
                    "증상을 조금 더 알려주시면 더 정확히 좁혀드릴 수 있어요."
                ),
                "sources": [], "is_followup": True,
            }

        if len(ranked) >= AMBIGUITY_MIN_DISEASES:
            # 되묻기 한도까지 다 써도 후보가 안 좁혀진 경우: 하나로 확정하지 않고
            # 후보 병명들을 나란히 보여주는 정도면 충분한, 사실상 정해진 답변이다.
            # 예전엔 이 문구를 LLM한테 "이렇게 표현해서 답해라"는 지시문(multi_note)으로
            # 넘겨서 다듬어 답하게 했는데, 로컬 8B 모델이 그 지시문 자체를 그대로
            # 베껴서 답변에 넣어버리는 문제가 실측으로 확인됐다(지시 이행 신뢰도 문제 —
            # RAGAS 로컬 judge 불안정성과 같은 계열의 한계). 내용이 이미 정해져 있으니
            # "10개 초과"/"3~10개" 분기와 동일하게 LLM 호출 없이 바로 답한다.
            print(f"  되묻기 한도 도달, 여전히 애매함 -> 후보: {candidate_names}")
            collect_sources([v["meta"] for v in ranked], sources)
            # candidate_names는 투표수·거리 기준으로 이미 정렬되어 나오므로,
            # 앞의 3개가 가장 유력한 후보다.
            total_count = len(candidate_names)
            top3 = candidate_names[:3]
            top3_str = ", ".join(top3)
            answer = (
                f"총 {total_count}개의 질환이 예상되며, 이 중 가장 의심되는 질환은 "
                f"{top3_str}입니다. 하나로 확정하기는 어려우니, 증상이 지속되거나 "
                "심해지면 가까운 병원에서 진료를 받아보시길 권장합니다.\n\n"
                "※ 참고용 안내이며 진단이 아닙니다. 정확한 진단은 병원에서 받으셔야 합니다."
            )
            return {"answer": answer, "sources": sources, "is_followup": False}
        else:
            # 후보가 1~2개로 충분히 좁혀진 경우: 그 병(들)에 대한 모든 증상별
            # 검색 결과(pool)를 모아 거리순 top-3 청크로 답변 컨텍스트를 만든다
            # (병명별 최고 청크 1개씩만 있는 ranked보다 내용이 풍부함).
            top_diseases = set(candidate_names)
            matched = [(doc, dist, meta) for doc, dist, meta in pool if meta.get("disease") in top_diseases]
            matched.sort(key=lambda x: x[1])
            docs = [m[0] for m in matched[:3]]
            metadatas = [m[2] for m in matched[:3]]
            context = build_context(docs, metadatas)
            collect_sources(metadatas, sources)
            multi_note = (
                "가능성이 있는 질환은 단정하지 말고 '~일 가능성이 있습니다', '~이 의심됩니다'처럼 "
                "추정하는 표현으로 안내하세요. 진료과도 '~에서 진료를 받을 수 있습니다'처럼 사실을 "
                "서술하지 말고, '~과 방문을 권장드립니다'처럼 추천하는 어투로 안내하세요. "
                "이건 참고용 안내일 뿐 진단이 아니라는 것과, 정확한 진단은 병원에서 받아야 한다는 것을 안내하세요.\n"
            )
    else:
        # 2. 병명이 하나 이상 감지된 경우: 병명별로 검색 범위를 좁혀서 각각 답변용 컨텍스트를 만든다.
        n_results = 3 if len(diseases) == 1 else 2
        sections = []
        for disease in diseases:
            docs, distances, metadatas = search_disease(query_embedding, disease, category_keywords, n_results)
            print(f"  - {disease}: 검색된 문서 수={len(docs)}, 거리={distances}")
            if not docs or distances[0] > THRESHOLD:
                sections.append(f"### {disease}\n관련 정보를 찾지 못했습니다.")
                continue
            docs, distances, metadatas = rerank(user_input, docs, distances, metadatas, n_results)
            sections.append(f"### {disease}\n" + build_context(docs, metadatas))
            collect_sources(metadatas, sources)

        context = "\n\n".join(sections)
        multi_note = (
            "아래는 서로 다른 질병에 대한 정보입니다. 병명별로 항목을 나누어 각각 답변하세요.\n"
            if len(diseases) > 1 else ""
        )

    # 3. llama-cpp-python으로 로컬 추론
    # CPU 추론이라 시간이 걸릴 수 있으니 이벤트 루프를 막지 않도록 스레드에서 실행한다.
    user_message = f"[의학 지식]:\n{context}\n\n{multi_note}질문: {user_input}"

    try:
        # get_llm()도 스레드 안에서 호출해야 한다 — 첫 호출이면 8B 모델 로딩(100초
        # 이상)이 여기서 일어나는데, await 밖(코루틴 본문)에서 부르면 그 시간 동안
        # 이벤트 루프 전체가 막혀버린다.
        response = await asyncio.to_thread(
            lambda: get_llm().create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 자가 진단 보조 AI입니다. 아래 제공된 [의학 지식]만을 바탕으로 "
                                    "사용자의 질문에 친절하고 명확하게 답변하세요. 지식에 없는 내용은 절대로 "
                                    "지어내지 마세요. 필요하면 병원 방문을 권유하세요.",
                    },
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1024,
            )
        )
        answer = response["choices"][0]["message"]["content"]
        return {"answer": answer, "sources": sources, "is_followup": False}
    except Exception as e:
        return {"answer": f"LLM 호출 오류: {str(e)}", "sources": sources, "is_followup": False}


@app.get("/", response_class=HTMLResponse)
async def index_page():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    # README에 "python main.py"로 실행하는 방법이 안내되어 있는데, 이 진입점이
    # 없으면 모델만 로딩하고 서버는 안 뜬 채로 스크립트가 그냥 끝나버린다
    # (실측으로 확인된 문제 — Uvicorn 시작 메시지 없이 바로 종료됨).
    # HOST/PORT를 환경변수로 읽어서, 로컬 실행(기본값 127.0.0.1:8000)은 그대로 두고
    # Hugging Face Spaces 같은 배포 환경(0.0.0.0, 포트 7860 등)도 코드 수정 없이 지원한다.
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)