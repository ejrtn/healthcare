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
RERANK_CANDIDATES = 10  # 벡터 검색으로 넉넉히 뽑아서 이 중 top_k만 리랭킹으로 추린다
ANSWER_TOP_K = 5         # 최종 답변에 쓸 컨텍스트 청크 수 (여러 후보 질환을 같이 언급할 수 있게 여유 있게)


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
# 주의: 이 값은 구 아키텍처(규칙 기반 라우팅)에서 실측으로 정한 값이다. 아래
# classify_intent()가 만드는 검색 문구는 예전과 특성이 달라질 수 있어(자유
# 문장 생성이라 더 풍부하거나 반대로 더 짧을 수 있음), 재측정이 필요할 수 있다.
THRESHOLD = 0.5

HISTORY_MESSAGE_LIMIT = 6  # LLM에게 넘길 최근 대화 턴 수 (너무 길면 컨텍스트/속도 부담)


def build_history_messages(history: List[dict]) -> List[dict]:
    """최근 history를 OpenAI 스타일 messages 리스트로 변환 (최근 N개만)."""
    return [
        {"role": h.get("role", "user"), "content": h.get("content", "")}
        for h in history[-HISTORY_MESSAGE_LIMIT:]
        if h.get("content")
    ]


def classify_intent(user_input: str, history: List[dict]) -> str:
    """
    새 아키텍처의 핵심: 규칙 기반 병명/카테고리 매칭을 전부 걷어내고, LLM에게 딱
    한 번만 묻는다 — "의료 질문이면 검색용 키워드를, 일상 대화면 'CASUAL'만 답해."

    === 왜 이게 예전 에이전틱 실험과 다르다고 판단했나 ===
    finetune/RESULTS.md의 에이전틱 실험에서 "필드 2개를 동시에 구조화된 형식으로
    뽑아내라"는 tool calling은 정확도 0%까지 떨어졌지만, "필드 1개만 추출"하는
    단순 작업은 3/3(100%)으로 훨씬 안정적이었다. 여기서 시키는 일("의료냐 아니냐
    + 검색어 하나")은 구조화 JSON이 아니라 자유 텍스트 한 줄이라, 그 "필드 1개"
    유형에 더 가깝다.

    === 그래도 남아있는 리스크 ===
    로컬 8B의 지시 이행 신뢰도가 낮다는 건 이 프로젝트에서 여러 번 실측으로
    확인된 사실이다(RAGAS 로컬 judge 40.6%, multi_note를 그대로 베껴 쓴 사례 등).
    그래서 출력이 이상하면(너무 길거나 줄바꿈이 섞이면) "CASUAL 아님"으로 임의
    판단하지 않고, 사용자의 원문 그대로를 검색어로 써서 최소한 답변이 아예
    누락되는 일은 없게 한다 — 최악의 경우 검색이 THRESHOLD에서 걸러져 안전하게
    "근거 없음" 폴백으로 빠질 뿐이다.
    """
    # Few-shot 예시를 실제 user/assistant 메시지 쌍으로 넣는다 — 규칙을 말로
    # 설명하는 것보다, chat 포맷 그대로 "이런 입력엔 이렇게 답한다"는 예시를
    # 직접 보여주는 쪽이 로컬 8B 같은 작은 모델에는 훨씬 잘 먹힌다(실측으로
    # 검증된 건 아니라 이번에 같이 확인해야 할 가설이다).
    FEW_SHOT_EXAMPLES = [
        {"role": "user", "content": "안녕하세요"},
        {"role": "assistant", "content": "CASUAL"},
        {"role": "user", "content": "너는 누구야?"},
        {"role": "assistant", "content": "CASUAL"},
        {"role": "user", "content": "고마워요 도움 됐어요"},
        {"role": "assistant", "content": "CASUAL"},
        {"role": "user", "content": "기침이 심해요"},
        {"role": "assistant", "content": "기침이 심함"},
        {"role": "user", "content": "당뇨병 치료법이 뭐야"},
        {"role": "assistant", "content": "당뇨병 치료"},
        {"role": "user", "content": "미열도 있어요"},  # 직전에 "기침이 심해요"가 있었다고 가정한 예시
        {"role": "assistant", "content": "기침이 심하고 미열도 있음"},
        # 대화 방향이 도중에 바뀌는 경우: 첫 메시지는 그냥 컨디션 하소연(잡담처럼
        # 보여서 CASUAL)이었다가, 다음 메시지에서 구체적인 신체 증상이 나오면
        # 그 시점부터 의료 질문으로 재해석하고 이전 맥락과 합친다. "몸 상태가
        # 안 좋다"는 말 자체는 애매하지만, 뒤이어 특정 증상이 나오면 그게
        # 힌트가 된다 — 알약 프로젝트에서 "투명하고 액체가 들어있으면
        # 연질캡슐"처럼 구체적 판별 기준을 줬을 때 정확도가 올라갔던 것과
        # 같은 원리로, 여기서도 "애매함 → 후속 메시지로 드러남" 패턴을
        # 구체적인 예시로 직접 보여준다.
        {"role": "user", "content": "요즘 컨디션이 안 좋네"},
        {"role": "assistant", "content": "CASUAL"},
        {"role": "user", "content": "화장실을 너무 자주 가서 그런가"},
        {"role": "assistant", "content": "컨디션이 안 좋고 화장실을 자주 감"},
    ]

    try:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "사용자의 마지막 메시지가 의료·건강·증상·질병에 관한 질문인지 판단하세요. "
                        "의료 관련이면 검색에 쓸 핵심 키워드(증상 설명 또는 병명)만 한 줄로 답하세요 "
                        "— 이전 대화에서 언급된 증상이 있으면 이번 메시지와 자연스럽게 합쳐서 하나의 "
                        "검색 문장으로 만드세요. 이전 메시지가 애매하거나 잡담처럼 보였더라도, 이번 "
                        "메시지에서 구체적인 몸 상태·증상이 나오면 그 시점부터 의료 질문으로 판단하고 "
                        "이전 맥락과 합치세요. 의료와 무관한 인사·잡담이면 정확히 'CASUAL'이라고만 "
                        "답하세요. 다른 설명이나 따옴표는 절대 추가하지 마세요. 아래는 예시입니다."
                    ),
                },
                *FEW_SHOT_EXAMPLES,
                *build_history_messages(history),
                {"role": "user", "content": user_input},
            ],
            max_tokens=64,
        )
        result = response["choices"][0]["message"]["content"].strip().strip('"').strip("'")
    except Exception as e:
        print(f"  [classify_intent 오류] {e} -> 원문을 검색어로 사용")
        return user_input

    if not result or "\n" in result or len(result) > 100:
        print(f"  [classify_intent 이상 출력] {result!r} -> 원문을 검색어로 사용")
        return user_input
    return result


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

    1등만 THRESHOLD를 넘는지 보는 건 "상대 평가"라 후보가 부족한 경우 뒤쪽 몇 개는
    실제로는 THRESHOLD를 넘는(관련 없는) 것들로 채워질 수 있다. 그런 약한 후보까지
    리랭커에게 넘기면 엉뚱하게 승격시킬 수 있으므로, 리랭킹 전에 후보군 자체를
    절대 기준(THRESHOLD)으로 한 번 걸러낸다.
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
    """전체 컬렉션(병명/카테고리 필터 없음)에서 의미상 가장 가까운 후보를 뽑는다."""
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )
    return results["documents"][0], results["distances"][0], results["metadatas"][0]


@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    print(f"질문: {user_input}")

    # ── 1. 의도 판단: 의료 질문인가, 그냥 일상 대화인가 ──────────
    # (LLM 호출이라 CPU 추론 기준 시간이 걸릴 수 있어 스레드에서 실행)
    intent = await asyncio.to_thread(classify_intent, user_input, request.history)
    print(f"  의도 판단 결과: {intent!r}")

    if intent.strip().upper() == "CASUAL":
        # ── 2a. 일상 대화 — 검색 없이 LLM이 대화 이력을 참고해서 자유롭게 답한다 ──
        try:
            response = await asyncio.to_thread(
                llm.create_chat_completion,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 친근한 자가 진단 보조 AI입니다. 의료와 무관한 가벼운 "
                            "인사나 잡담에는 짧고 자연스럽게 답하세요. 의료 질문이 아니라는 "
                            "판단이 맞다면, 필요시 '증상이나 궁금한 건강 정보를 말씀해주시면 "
                            "도와드릴게요' 정도로 안내해도 좋습니다."
                        ),
                    },
                    *build_history_messages(request.history),
                    {"role": "user", "content": user_input},
                ],
                max_tokens=512,
            )
            answer = response["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"LLM 호출 오류: {str(e)}"
        return {"answer": answer, "sources": [], "is_followup": False}

    # ── 2b. 의료 질문 — intent(검색 키워드)로 전체 컬렉션 벡터 검색 ──
    query_embedding = embed_model.encode([intent]).tolist()
    docs, distances, metadatas = vector_search(query_embedding, n_results=RERANK_CANDIDATES)

    if not docs or distances[0] > THRESHOLD:
        return {
            "answer": (
                "죄송합니다. 입력하신 내용에 대해 참고할 수 있는 의학적 근거가 "
                "데이터베이스에 없습니다. 정확한 진단을 위해 가까운 병원 방문을 권장합니다."
            ),
            "sources": [], "is_followup": False,
        }

    docs, distances, metadatas = rerank(intent, docs, distances, metadatas, ANSWER_TOP_K)
    context = build_context(docs, metadatas)
    sources: List[dict] = []
    collect_sources(metadatas, sources)

    # ── 3. llama-cpp-python으로 로컬 추론 ──
    # 여러 질환이 섞여 나올 수 있으니, 하나로 단정하지 말고 가능성 위주로 안내하며
    # 필요하면 되물어보라고 시스템 프롬프트에서 직접 유도한다 (예전처럼 후보 개수를
    # 규칙으로 세서 분기하는 대신, 이 판단 자체를 LLM에게 맡기는 실험이다).
    try:
        response = await asyncio.to_thread(
            llm.create_chat_completion,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 자가 진단 보조 AI입니다. 아래 제공된 [의학 지식]만을 바탕으로 "
                        "사용자의 질문에 친절하고 명확하게 답변하세요. 지식에 없는 내용은 절대로 "
                        "지어내지 마세요. [의학 지식]에 서로 다른 질환이 여러 개 섞여 있다면 "
                        "하나로 단정하지 말고 '~일 가능성이 있습니다'처럼 안내하고, 증상을 더 "
                        "물어봐야 좁혀질 것 같으면 되물어보세요. 진료과 추천은 '~과 방문을 "
                        "권장드립니다'처럼 추천 어투로, 참고용 안내일 뿐 진단이 아니라는 점을 "
                        "함께 안내하세요."
                    ),
                },
                *build_history_messages(request.history),
                {"role": "user", "content": f"[의학 지식]:\n{context}\n\n질문: {user_input}"},
            ],
            max_tokens=1024,
        )
        answer = response["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"LLM 호출 오류: {str(e)}"

    return {"answer": answer, "sources": sources, "is_followup": False}


@app.get("/", response_class=HTMLResponse)
async def index_page():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
