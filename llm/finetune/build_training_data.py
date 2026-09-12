"""
eval/ground_truth.py의 정답 라벨(질문, 병명, 기대 카테고리)로부터
임베딩 모델 파인튜닝용 (anchor, positive, negative) triplet을 만든다.

- anchor: 질문 그대로 (main.py가 질의 임베딩할 때 쓰는 형식과 동일)
- positive: 기대 카테고리 중 하나에 해당하는 실제 청크 ("병명 카테고리: 내용" 형식,
  main.py가 문서를 임베딩할 때 쓰는 형식과 동일)
- negative: 같은 병의 "기대하지 않는" 다른 카테고리 청크 (하드 네거티브 —
  실제로 헷갈렸던 것과 같은 종류의 오답)

일반화(generalization)를 확인하려면 학습에 안 쓴 데이터로 평가해야 하므로,
질문 단위로 train/test를 나눈다 (같은 질문이 양쪽에 안 걸치게).
"""
import os
import sys
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "eval"))
sys.path.insert(0, PROJECT_ROOT)

from ground_truth import GROUND_TRUTH  # noqa: E402

RANDOM_SEED = 42
TEST_RATIO = 0.25


def load_chunks_by_disease_category():
    """병명+카테고리 -> 그 카테고리에 속한 청크 content 리스트."""
    lookup: dict = {}
    path = os.path.join(PROJECT_ROOT, "data", "medical_knowledge.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["disease"], r["category"])
            lookup.setdefault(key, []).append(r["content"])
    return lookup


def build_triplets():
    lookup = load_chunks_by_disease_category()
    rng = random.Random(RANDOM_SEED)

    triplets = []
    skipped = []
    for question, disease, expected in GROUND_TRUTH:
        # positive: 기대 카테고리 중 실제로 청크가 있는 것 하나
        positive_categories = [c for c in expected if (disease, c) in lookup]
        if not positive_categories:
            skipped.append((question, "positive 카테고리 청크 없음"))
            continue
        pos_category = rng.choice(positive_categories)
        pos_content = rng.choice(lookup[(disease, pos_category)])

        # negative: 같은 병의 "기대하지 않는" 다른 카테고리 청크 (하드 네거티브)
        other_categories = [
            cat for (dis, cat) in lookup.keys()
            if dis == disease and cat not in expected
        ]
        if not other_categories:
            skipped.append((question, "negative 카테고리 없음"))
            continue
        neg_category = rng.choice(other_categories)
        neg_content = rng.choice(lookup[(disease, neg_category)])

        triplets.append({
            "question": question,
            "disease": disease,
            "anchor": question,
            "positive": f"{disease} {pos_category}: {pos_content}",
            "negative": f"{disease} {neg_category}: {neg_content}",
            "pos_category": pos_category,
            "neg_category": neg_category,
        })

    if skipped:
        print(f"스킵된 문항 {len(skipped)}건: {skipped}")

    rng.shuffle(triplets)
    n_test = max(1, int(len(triplets) * TEST_RATIO))
    test_set = triplets[:n_test]
    train_set = triplets[n_test:]
    return train_set, test_set


if __name__ == "__main__":
    train_set, test_set = build_triplets()
    print(f"전체 triplet: {len(train_set) + len(test_set)}건 -> train {len(train_set)} / test {len(test_set)}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    print("finetune/train.json, finetune/test.json 저장 완료")
