import os
import sys
import json
import asyncio
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
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 2. ChromaDB 영구 저장 클라이언트 (vector_db 폴더에 디스크 저장)
chroma_client = chromadb.PersistentClient(path="vector_db")
collection = chroma_client.get_or_create_collection(
    name="medical_knowledge_v2",  # 청킹 스키마 변경으로 컬렉션명 갱신 (기존 컬렉션과 분리)
    metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
)

# 2-1. LLM (Ollama 대신 llama-cpp-python으로 gguf 파일을 직접 로드)
# n_threads를 지정하지 않으면 llama-cpp-python이 보수적인 기본값을 쓰는 경우가 있어
# 물리 코어 수에 맞춰 명시적으로 지정 (환경에 맞게 os.cpu_count() 등으로 조정 가능)
CPU_THREADS = 14
llm = Llama(
    model_path="models/meta-llama-3.1-8b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=CPU_THREADS,        # 응답 생성(decode) 병렬도
    n_threads_batch=CPU_THREADS,  # 프롬프트 처리(prefill) 병렬도
    n_gpu_layers=0,  # GPU(CUDA build)로 가속하려면 늘리세요 (예: 20~35)
    chat_format="llama-3",
    verbose=False,
)

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
        response = llm.create_chat_completion(
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


# 병명 없이 증상만 설명하는 질문에서, 후보 질환들의 거리가 다 비슷하게 가까우면
# 하나로 단정하지 않고 되물어서 좁혀나간다.
AMBIGUITY_MARGIN = 0.08       # 1등 거리 + 이 값 안에 있으면 "비슷하게 가까운" 후보로 침
AMBIGUITY_MIN_DISEASES = 3    # 서로 다른 병이 이 개수 이상이면 애매하다고 판단
AMBIGUITY_MAX_DISEASES = 10   # 이 개수를 넘으면 목록을 보여주지 않고 증상을 더 물어본다
MAX_FOLLOWUP_ROUNDS = 3       # 후보가 안 좁혀져도 무한정 되묻지 않도록 하는 안전장치(주 기준은 후보 개수)
# 병명 미언급 질문에서 뽑아올 후보 청크 수. "기침"처럼 흔한 증상은 원인이 10개를
# 넘을 수 있어서(실측: 20개→7병, 30개→8병, 50개→12병) 넉넉히 40으로 잡았다.
SYMPTOM_CANDIDATES = 40


def build_symptom_query(message: str, history: List[dict]) -> str:
    """지금까지 대화에서 사용자가 말한 증상 설명을 이어붙여 누적 질의문을 만든다."""
    parts = [h.get("content", "") for h in history if h.get("role") == "user"]
    parts.append(message)
    return " ".join(p for p in parts if p)


def find_ambiguous_candidates(distances, metadatas) -> List[dict]:
    """상위 후보 중 1등과 거리가 비슷한 서로 다른 병들을 뽑는다 (많으면 애매한 질문)."""
    if not distances:
        return []
    top_distance = distances[0]
    seen: dict = {}
    for dist, meta in zip(distances, metadatas):
        if dist > top_distance + AMBIGUITY_MARGIN:
            continue
        disease = meta.get("disease")
        if disease not in seen:
            seen[disease] = meta
    return list(seen.values())


def pick_best_chunk_per_disease(candidate_diseases: List[str], docs, distances, metadatas):
    """
    되묻기 한도까지 다 썼는데도 후보가 안 좁혀졌을 때, 후보 병명별로 가장 가까운
    청크를 하나씩만 골라 컨텍스트를 만든다 (한 병에 쏠리지 않고 후보들을 골고루 보여주기 위함).
    """
    best: dict = {}
    for doc, dist, meta in zip(docs, distances, metadatas):
        disease = meta.get("disease")
        if disease in candidate_diseases and (disease not in best or dist < best[disease][0]):
            best[disease] = (dist, doc, meta)
    picked = sorted(best.values(), key=lambda x: x[0])
    return [p[1] for p in picked], [p[2] for p in picked]


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
        # 지금까지 대화에서 나온 증상 설명을 다 이어붙여서 검색한다 (멀티턴 누적).
        symptom_query = build_symptom_query(user_input, request.history)
        print(f"  누적 증상 질의: {symptom_query!r} (history {len(request.history)}턴)")
        symptom_embedding = embed_model.encode([symptom_query]).tolist()

        # 임계값 판단은 리랭킹 전, 벡터 검색의 원래 1등(코사인 거리가 가장 가까운 것)으로 한다.
        # 리랭킹은 "관련 있는 것들 중에 뭘 보여줄지"를 정하는 역할이지,
        # "관련이 있긴 한지"를 판단하는 역할이 아니기 때문이다.
        docs, distances, metadatas = vector_search(symptom_embedding, n_results=SYMPTOM_CANDIDATES)

        if not docs or distances[0] > THRESHOLD:
            # 규칙 기반 병명 매칭도, 임베딩 검색도 둘 다 실패한 경우 — "디스크"처럼
            # 등록된 정식 명칭("추간판탈출증(디스크)")의 일부만 말한 구어체/줄임말일
            # 가능성이 있으니, 완전히 포기하기 전에 LLM에게 한 번 물어봐서 맞으면
            # 되물어보고, 그마저도 안 되면 원래 폴백 메시지를 그대로 낸다.
            guess = guess_standard_term(user_input)
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
        candidates = find_ambiguous_candidates(distances, metadatas)
        candidate_names = [c.get("disease") for c in candidates]

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

        if len(candidates) >= AMBIGUITY_MIN_DISEASES:
            # 되묻기 한도까지 다 써도 후보가 안 좁혀진 경우: 하나로 확정하지 않고
            # 후보 병명들을 나란히 보여주는 정도면 충분한, 사실상 정해진 답변이다.
            # 예전엔 이 문구를 LLM한테 "이렇게 표현해서 답해라"는 지시문(multi_note)으로
            # 넘겨서 다듬어 답하게 했는데, 로컬 8B 모델이 그 지시문 자체를 그대로
            # 베껴서 답변에 넣어버리는 문제가 실측으로 확인됐다(지시 이행 신뢰도 문제 —
            # RAGAS 로컬 judge 불안정성과 같은 계열의 한계). 내용이 이미 정해져 있으니
            # "10개 초과"/"3~10개" 분기와 동일하게 LLM 호출 없이 바로 답한다.
            candidate_names = [c.get("disease") for c in candidates]
            print(f"  되묻기 한도 도달, 여전히 애매함 -> 후보: {candidate_names}")
            docs, metadatas = pick_best_chunk_per_disease(candidate_names, docs, distances, metadatas)
            collect_sources(metadatas, sources)
            # candidate_names는 find_ambiguous_candidates()에서 벡터 검색 거리순(가까운
            # 순서)으로 이미 정렬되어 나오므로, 앞의 3개가 가장 유력한 후보다.
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
            # 후보가 충분히 좁혀진 경우: 기존처럼 top-3로 좀 더 구체적으로 답한다.
            docs, distances, metadatas = rerank(symptom_query, docs, distances, metadatas, 3)
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
        response = await asyncio.to_thread(
            llm.create_chat_completion,
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
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)