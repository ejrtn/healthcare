"""
청크 크기(chunk_size)를 100~600자로 바꿔가며 카테고리 라우팅 정확도를 비교하는 실험.

ground_truth.py에 나오는 6개 병명(고혈압/당뇨병/감기/간흡충증/골다공증/두통)만 골라서
청크 크기별로 별도 임시 ChromaDB 컬렉션을 만들고, 리랭킹 없이 벡터 검색 1등이
정답 카테고리와 일치하는지만 채점한다 (리랭킹 효과는 run_eval.py가 따로 담당).

실제 프로젝트의 data/medical_knowledge.jsonl, vector_db/는 전혀 건드리지 않고
eval/.tmp/ 아래에만 임시 파일을 만든다.

실행:
    cd project
    python eval/chunk_size_experiment.py
"""
import os
import sys
import json
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(EVAL_DIR, ".tmp")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, EVAL_DIR)

import chromadb  # noqa: E402
import parse_xml  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from ground_truth import GROUND_TRUTH  # noqa: E402

CHUNK_SIZES = [100, 200, 300, 400, 500, 600]
OVERLAP = 60
TARGET_DISEASES = {d for _, d, _ in GROUND_TRUTH}


def load_disease_categories(chunk_size: int, jsonl_path: str):
    """청크 크기와 무관하게, 이번 실험용으로 새로 뽑은 카테고리 목록을 읽는다."""
    cats = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["disease"] in TARGET_DISEASES:
                cats.setdefault(r["disease"], [])
                if r["category"] not in cats[r["disease"]]:
                    cats[r["disease"]].append(r["category"])
    return cats


def resolve_categories(disease_categories, disease, keywords):
    if not keywords:
        return []
    actual = disease_categories.get(disease, [])
    return [c for c in actual if any(k in c for k in keywords)]


def run():
    os.makedirs(TMP_DIR, exist_ok=True)
    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # detect_category_keywords는 main.py 것과 완전히 동일한 로직을 써야 공정한 비교가 되므로 그대로 가져온다.
    import main  # noqa: E402  (임포트 시점에 LLM까지 로드되어 느리지만, 로직 재사용을 위해 감수)

    client = chromadb.PersistentClient(path=os.path.join(TMP_DIR, "vector_db"))

    summary = []

    for size in CHUNK_SIZES:
        jsonl_path = os.path.join(TMP_DIR, f"mk_{size}.jsonl")
        dis_path = os.path.join(TMP_DIR, f"dis_{size}.json")
        cat_path = os.path.join(TMP_DIR, f"cat_{size}.json")

        parse_xml.parse_all_xml(
            os.path.join(PROJECT_ROOT, "data", "kdca_health_info"),
            jsonl_path, dis_path, cat_path,
            chunk_size=size, overlap=OVERLAP,
        )

        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["disease"] in TARGET_DISEASES:
                    records.append(r)

        disease_categories = load_disease_categories(size, jsonl_path)

        try:
            client.delete_collection("chunk_experiment")
        except Exception:
            pass
        collection = client.create_collection("chunk_experiment", metadata={"hnsw:space": "cosine"})

        docs = [r["content"] for r in records]
        embed_texts = [f"{r['disease']} {r['category']}: {r['content']}" for r in records]
        embeddings = embed_model.encode(embed_texts).tolist()
        ids = [f"doc_{i}" for i in range(len(records))]
        metas = [{"disease": r["disease"], "category": r["category"]} for r in records]
        collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)

        correct = 0
        detail = []
        for question, disease, expected in GROUND_TRUTH:
            keywords = main.detect_category_keywords(question)
            where = {"disease": disease}
            resolved = resolve_categories(disease_categories, disease, keywords)
            if resolved:
                where = {"$and": [{"disease": disease}, {"category": {"$in": resolved}}]}

            query_embedding = embed_model.encode([question]).tolist()
            res = collection.query(query_embeddings=query_embedding, n_results=3, where=where, include=["metadatas"])
            top_category = res["metadatas"][0][0].get("category") if res["metadatas"][0] else None
            ok = top_category in expected
            correct += ok
            detail.append({"question": question, "expected": expected, "top": top_category, "ok": ok})

        summary.append({
            "chunk_size": size,
            "chunk_count": len(records),
            "correct": correct,
            "total": len(GROUND_TRUTH),
            "detail": detail,
        })
        print(f"[chunk_size={size}] 청크수={len(records)} 정확도={correct}/{len(GROUND_TRUTH)}")
        for d in detail:
            if not d["ok"]:
                print(f"    틀림: {d['question']} (기대={d['expected']}, 실제={d['top']})")

    print()
    print("=== 요약 ===")
    for s in summary:
        print(f"{s['chunk_size']}자: {s['correct']}/{s['total']} (청크수 {s['chunk_count']})")

    out_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chunk_size_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 {out_path}에 저장되었습니다.")

    # ChromaDB가 sqlite 파일 핸들을 계속 물고 있어서(Windows) 바로 지우면
    # vector_db 하위 폴더가 남을 수 있다. client 참조를 끊어서 최대한 정리해본다.
    del client
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    if os.path.exists(TMP_DIR):
        print(f"참고: {TMP_DIR} 일부가 남아있을 수 있습니다 (Windows 파일 잠금). 수동으로 지워도 안전합니다.")


if __name__ == "__main__":
    run()
