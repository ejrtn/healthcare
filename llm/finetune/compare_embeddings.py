"""
파인튜닝 전(ko-sroberta-multitask) vs 후(ko-sroberta-finetuned)를 비교한다.

측정 방법: 각 triplet(anchor, positive, negative)에 대해
cos_sim(anchor, positive) > cos_sim(anchor, negative) 인지를 체크한다.
이게 정확히 TripletLoss가 최적화하는 대상이라, 파인튜닝 효과를 가장 직접적으로 보여준다.

train(학습에 쓴 25문항)과 test(학습에 안 쓴 8문항)를 나눠서 보는 게 핵심이다.
train만 좋아지고 test는 그대로거나 나빠지면 그냥 암기(overfit)한 것이고,
test까지 좋아져야 "일반화됐다"고 말할 수 있다.

실행:
    cd project
    python finetune/compare_embeddings.py
"""
import os
import json

from sentence_transformers import SentenceTransformer
import numpy as np

BASE_MODEL = "jhgan/ko-sroberta-multitask"
FINETUNED_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ko-sroberta-finetuned")


def cos_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate(model, triplets):
    correct = 0
    margins = []
    for t in triplets:
        anchor_emb = model.encode(t["anchor"])
        pos_emb = model.encode(t["positive"])
        neg_emb = model.encode(t["negative"])
        pos_sim = cos_sim(anchor_emb, pos_emb)
        neg_sim = cos_sim(anchor_emb, neg_emb)
        margins.append(pos_sim - neg_sim)
        if pos_sim > neg_sim:
            correct += 1
    return correct, len(triplets), sum(margins) / len(margins)


def main():
    finetune_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(finetune_dir, "train.json"), encoding="utf-8") as f:
        train_set = json.load(f)
    with open(os.path.join(finetune_dir, "test.json"), encoding="utf-8") as f:
        test_set = json.load(f)

    results = {}
    for name, path in [("원본(파인튜닝 전)", BASE_MODEL), ("파인튜닝 후", FINETUNED_MODEL)]:
        print(f"모델 로딩: {name} ({path})")
        model = SentenceTransformer(path)

        train_correct, train_total, train_margin = evaluate(model, train_set)
        test_correct, test_total, test_margin = evaluate(model, test_set)

        results[name] = {
            "train": {"correct": train_correct, "total": train_total, "avg_margin": train_margin},
            "test": {"correct": test_correct, "total": test_total, "avg_margin": test_margin},
        }
        print(f"  train(학습에 쓴 데이터): {train_correct}/{train_total}, 평균 마진(positive-negative 유사도 차)={train_margin:.4f}")
        print(f"  test(안 보여준 데이터):  {test_correct}/{test_total}, 평균 마진={test_margin:.4f}")
        print()

    out_path = os.path.join(finetune_dir, "comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과가 {out_path}에 저장되었습니다.")


if __name__ == "__main__":
    main()
