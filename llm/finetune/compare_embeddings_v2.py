"""
원본 vs v2(카테고리+증상 합쳐서 재학습) 임베딩 모델을 비교한다.

1. train_v2/test_v2 triplet 정확도 — "카테고리 구분"과 "증상 기반 감별진단"을
   타입별로 나눠서 본다 (증상 데이터를 추가한 게 실제로 증상 쪽에도 도움이
   됐는지 확인하기 위함).
2. 거절(reject) 안전성 체크 — 파인튜닝이 "관련 있는 것들을 가깝게" 만드는
   과정이라, 자칫 전체적으로 다 가까워져서 무관한 질문까지 오염시킬 위험이
   있다. 무관한 질문 5개와 청크 샘플의 최대 유사도가 파인튜닝 후에도 여전히
   낮게 유지되는지 확인한다.

실행:
    cd project
    python finetune/compare_embeddings_v2.py
"""
import os
import sys
import json
import random

from sentence_transformers import SentenceTransformer
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "eval"))

from ground_truth import REJECT_CASES  # noqa: E402

BASE_MODEL = "jhgan/ko-sroberta-multitask"
FINETUNED_V2 = os.path.join(FINETUNE_DIR, "ko-sroberta-finetuned-v2")


def cos_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate_triplets(model, triplets):
    """전체 + (question이 SYMPTOM_CASES 문항인지로 대충 구분한) 타입별 정확도."""
    from ground_truth import SYMPTOM_CASES
    symptom_questions = {q for q, _ in SYMPTOM_CASES}

    results = {"all": [0, 0], "category": [0, 0], "symptom": [0, 0]}
    for t in triplets:
        anchor_emb = model.encode(t["anchor"])
        pos_emb = model.encode(t["positive"])
        neg_emb = model.encode(t["negative"])
        ok = cos_sim(anchor_emb, pos_emb) > cos_sim(anchor_emb, neg_emb)

        kind = "symptom" if t["question"] in symptom_questions else "category"
        for key in ("all", kind):
            results[key][0] += ok
            results[key][1] += 1
    return results


def evaluate_reject_safety(model, sample_size=200):
    """무관한 질문들이 무작위 청크 샘플과 얼마나 유사해지는지(낮아야 안전) 확인."""
    records = []
    path = os.path.join(PROJECT_ROOT, "data", "medical_knowledge.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    rng = random.Random(3)
    sample = rng.sample(records, sample_size)
    sample_embeds = [model.encode(f"{r['disease']} {r['category']}: {r['content']}") for r in sample]

    max_sims = []
    for question in REJECT_CASES:
        q_emb = model.encode(question)
        sims = [cos_sim(q_emb, s) for s in sample_embeds]
        max_sims.append(max(sims))
    return max_sims


def main():
    with open(os.path.join(FINETUNE_DIR, "train_v2.json"), encoding="utf-8") as f:
        train_v2 = json.load(f)
    with open(os.path.join(FINETUNE_DIR, "test_v2.json"), encoding="utf-8") as f:
        test_v2 = json.load(f)

    for name, path in [("원본", BASE_MODEL), ("v2(카테고리+증상)", FINETUNED_V2)]:
        print(f"=== {name} ({path}) ===")
        model = SentenceTransformer(path)

        train_res = evaluate_triplets(model, train_v2)
        test_res = evaluate_triplets(model, test_v2)

        for split_name, res in [("train", train_res), ("test", test_res)]:
            for kind in ("all", "category", "symptom"):
                c, t = res[kind]
                if t == 0:
                    continue
                print(f"  {split_name}/{kind}: {c}/{t} ({c/t:.1%})")

        max_sims = evaluate_reject_safety(model)
        # ChromaDB의 cosine 거리 = 1 - 코사인 유사도. THRESHOLD=0.5(거리) 통과를
        # 유사도로 바꾸면 "유사도 0.5 이상"이 위험권(거절 안 됨), 그 밑이 안전권.
        print(f"  거절 질문 5개의 (샘플 200개 청크 대비) 최대 유사도: {[f'{s:.3f}' for s in max_sims]}")
        print(f"  그중 최댓값: {max(max_sims):.3f} (0.5 이상이면 THRESHOLD를 못 넘겨 위험)")
        print()


if __name__ == "__main__":
    main()
