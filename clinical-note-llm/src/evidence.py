"""근거 검색 (RAG): 이번 요청의 대화에서 기록 문장의 근거가 될 발화를 찾는다.

검색 대상은 이번 진료의 대화뿐이다. 의학 지식(약 용량 사전 등)은 찾지 않는다 — 의사가 말하지 않은 값을
채워 넣는 것은 막으려는 지어내기와 같기 때문이다. 대화 하나는 발화 100개 안팎이라 벡터 DB 없이 요청마다
그 자리에서 검색한다.

- BM25: 약·검사 이름처럼 정확한 단어에 강하다.
- 임베딩(bge-small-en): "follow up in 4 weeks" ↔ "come back in a month"처럼 다르게 말한 것에 강하다.
- hybrid: 두 순위를 RRF로 합친다.
숫자 단어("one thousand")는 검색 전에 숫자로도 색인해 "1000"으로 찾을 수 있게 한다.
"""
import re

from faithfulness import spoken_numbers

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge 검색용 질의 접두어
MAX_WORDS = 60  # 긴 발화는 이 길이로 나눠 근거 위치를 좁힌다
STOPWORDS = set("a an the and or of to in on for with at by is are was were be been it this that you your i "
                "we he she they his her my me so um uh okay yeah".split())

_embedder = None


def split_turns(dialogue: str) -> list[dict]:
    """대화 → 발화 목록 [{"turn": 번호, "text": ...}]. 번호는 근거 인용에 쓴다."""
    turns = []
    for chunk in re.split(r"\n+|(?=\[(?:doctor|patient|patient_guest|guest_\w+)\])", dialogue):
        words = chunk.split()
        for i in range(0, len(words), MAX_WORDS):
            text = " ".join(words[i:i + MAX_WORDS])
            if text:
                turns.append({"turn": len(turns), "text": text})
    return turns


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower()) if t not in STOPWORDS]


def index_text(text: str) -> str:
    """색인용 텍스트: 숫자 단어를 숫자로도 덧붙인다 ("one thousand milligrams" → ... 1000)."""
    extra = sorted(n for n in spoken_numbers(text) if str(n) not in text)
    return text + (" " + " ".join(map(str, extra)) if extra else "")


def rank_bm25(query: str, turns: list[dict]) -> list[int]:
    from rank_bm25 import BM25Okapi
    docs = [tokens(index_text(t["text"])) or ["_"] for t in turns]
    scores = BM25Okapi(docs).get_scores(tokens(query))
    return sorted(range(len(turns)), key=lambda i: -scores[i])


def rank_embedding(query: str, turns: list[dict]) -> list[int]:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    q = _embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    d = _embedder.encode([t["text"] for t in turns], normalize_embeddings=True)
    scores = (d @ q.T).ravel()
    return sorted(range(len(turns)), key=lambda i: -scores[i])


def rank_hybrid(query: str, turns: list[dict], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: 각 방식의 순위 r에 1/(k + r)을 더한다. 점수 단위가 달라도 합칠 수 있다."""
    fused = [0.0] * len(turns)
    for ranking in (rank_bm25(query, turns), rank_embedding(query, turns)):
        for r, i in enumerate(ranking):
            fused[i] += 1 / (k + r)
    return sorted(range(len(turns)), key=lambda i: -fused[i])


RANKERS = {"bm25": rank_bm25, "embedding": rank_embedding, "hybrid": rank_hybrid}


def search(query: str, turns: list[dict], method: str = "hybrid", top_k: int = 3) -> list[dict]:
    if not turns or not query.strip():
        return []
    return [turns[i] for i in RANKERS[method](query, turns)[:top_k]]
