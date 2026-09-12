"""
build_training_data.py(병명+카테고리 33건, negative=같은 병 다른 카테고리)와
build_symptom_triplets.py(증상 기반 6건, negative=다른 병)를 합쳐서 train/test를
다시 나눈다. "같은 병 카테고리 구분"과 "증상으로 다른 병 구분" 두 능력을 같이
학습시키기 위함.

실행 순서:
    python finetune/build_training_data.py     # train.json/test.json (33건 기반, 참고용으로 남겨둠)
    python finetune/build_symptom_triplets.py  # symptom_triplets.json (6건)
    python finetune/build_training_data_v2.py  # 위 둘을 합쳐 train_v2.json/test_v2.json 생성
"""
import os
import json
import random

FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 42
TEST_RATIO = 0.25


def main():
    with open(os.path.join(FINETUNE_DIR, "train.json"), encoding="utf-8") as f:
        category_train = json.load(f)
    with open(os.path.join(FINETUNE_DIR, "test.json"), encoding="utf-8") as f:
        category_test = json.load(f)
    with open(os.path.join(FINETUNE_DIR, "symptom_triplets.json"), encoding="utf-8") as f:
        symptom_triplets = json.load(f)

    all_triplets = category_train + category_test + symptom_triplets
    print(f"카테고리 기반 {len(category_train) + len(category_test)}건 + 증상 기반 {len(symptom_triplets)}건 "
          f"= 총 {len(all_triplets)}건")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_triplets)
    n_test = max(1, int(len(all_triplets) * TEST_RATIO))
    test_set = all_triplets[:n_test]
    train_set = all_triplets[n_test:]

    print(f"train {len(train_set)} / test {len(test_set)}")
    with open(os.path.join(FINETUNE_DIR, "train_v2.json"), "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(os.path.join(FINETUNE_DIR, "test_v2.json"), "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    print("finetune/train_v2.json, finetune/test_v2.json 저장 완료")


if __name__ == "__main__":
    main()
