"""
eval/ground_truth.py의 SYMPTOM_CASES(증상 설명, 후보 병명 목록)로부터 임베딩
파인튜닝용 triplet을 추가로 만든다.

기존 build_training_data.py는 항상 "같은 병의 다른 카테고리"를 negative로 써서
"카테고리 구분"만 학습시켰다. 이 스크립트는 반대로 "다른 병"을 negative로 써서
"증상 설명으로 여러 병 중 맞는 걸 찾기"(감별진단) 능력을 학습시킨다.

- anchor: 증상 설명 그대로
- positive: acceptable_diseases 중 하나의 실제 청크
- negative: acceptable_diseases에 없는 무작위 다른 병의 청크 (혼동 가능한 병일수록
  좋지만, 지금은 무작위 샘플링 — 완전 엉뚱한 병이라도 "이건 관련 없다"를 배우는
  신호는 된다)
"""
import os
import sys
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "eval"))

from ground_truth import SYMPTOM_CASES  # noqa: E402

RANDOM_SEED = 11


def load_chunks_by_disease():
    lookup: dict = {}
    path = os.path.join(PROJECT_ROOT, "data", "medical_knowledge.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lookup.setdefault(r["disease"], []).append((r["category"], r["content"]))
    return lookup


def build():
    lookup = load_chunks_by_disease()
    rng = random.Random(RANDOM_SEED)
    all_diseases = list(lookup.keys())

    triplets = []
    for question, acceptable in SYMPTOM_CASES:
        pos_candidates = [d for d in acceptable if d in lookup]
        if not pos_candidates:
            print(f"스킵: {question} (acceptable 병명 중 데이터에 있는 게 없음)")
            continue
        pos_disease = rng.choice(pos_candidates)
        pos_category, pos_content = rng.choice(lookup[pos_disease])

        neg_disease = rng.choice([d for d in all_diseases if d not in acceptable])
        neg_category, neg_content = rng.choice(lookup[neg_disease])

        triplets.append({
            "question": question,
            "anchor": question,
            "positive": f"{pos_disease} {pos_category}: {pos_content}",
            "negative": f"{neg_disease} {neg_category}: {neg_content}",
            "pos_disease": pos_disease,
            "neg_disease": neg_disease,
        })

    return triplets


if __name__ == "__main__":
    triplets = build()
    print(f"증상 기반 triplet {len(triplets)}건 생성")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symptom_triplets.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"{out_path}에 저장했습니다.")
