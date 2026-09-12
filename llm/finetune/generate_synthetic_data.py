"""
수작업 라벨링(33문항) 없이 학습 데이터 규모를 키우기 위해, 이미 있는 청크
내용을 LLM에게 보여주고 "이 내용을 물어볼 법한 자연스러운 질문"을 생성시킨다.

- (병명, 카테고리) 조합을 다양한 병에 걸쳐 무작위로 샘플링 (한 병에 쏠리지 않게)
- 카테고리가 2개 이상인 병만 대상으로 함 (negative를 같은 병의 다른 카테고리에서
  뽑아야 하므로)
- 생성된 질문 + 정답 청크(positive) + 같은 병의 다른 카테고리 청크(negative)로
  triplet을 만든다

실행 (오래 걸림 — 샘플 수 x LLM 호출 1회씩):
    cd project
    python finetune/generate_synthetic_data.py
"""
import os
import sys
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import main  # noqa: E402  (llm 재사용을 위해 임포트 — 8B 모델 로딩에 시간 걸림)

RANDOM_SEED = 7
N_SAMPLES = 120


def load_records():
    recs = []
    path = os.path.join(PROJECT_ROOT, "data", "medical_knowledge.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["chunk_index"] == 0:  # 대표 청크(첫 청크)만 사용
                recs.append(r)
    return recs


def build_disease_category_pool(records):
    by_disease = {}
    for r in records:
        by_disease.setdefault(r["disease"], {})[r["category"]] = r["content"]
    # 카테고리가 2개 이상인 병만 (negative 뽑을 데가 있어야 하므로)
    return {d: cats for d, cats in by_disease.items() if len(cats) >= 2}


def generate_question(disease: str, category: str, content: str) -> str:
    """이 청크 내용을 물어볼 법한 자연스러운 한국어 질문 1개를 LLM이 생성."""
    snippet = content[:300]
    response = main.llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "당신은 의료 정보 챗봇 학습 데이터를 만드는 도우미입니다. "
                            "주어진 [내용]을 실제로 물어볼 법한 자연스러운 한국어 질문을 "
                            "정확히 1개만 만드세요. 질문에는 병명을 포함하되, [내용]에 있는 "
                            "문장을 그대로 베끼지 말고 사용자가 실제로 물어볼 법한 말투로 "
                            "바꾸세요. 질문 외의 다른 말은 하지 마세요.",
            },
            {"role": "user", "content": f"병명: {disease}\n카테고리: {category}\n내용: {snippet}"},
        ],
        max_tokens=80,
    )
    return response["choices"][0]["message"]["content"].strip().strip('"')


def main_run():
    records = load_records()
    pool = build_disease_category_pool(records)
    print(f"대상 병명 수(카테고리 2개 이상): {len(pool)}")

    rng = random.Random(RANDOM_SEED)
    diseases = list(pool.keys())
    rng.shuffle(diseases)

    synthetic = []
    for i, disease in enumerate(diseases):
        if len(synthetic) >= N_SAMPLES:
            break
        categories = list(pool[disease].keys())
        pos_category = rng.choice(categories)
        neg_category = rng.choice([c for c in categories if c != pos_category])

        question = generate_question(disease, pos_category, pool[disease][pos_category])
        synthetic.append({
            "question": question,
            "disease": disease,
            "anchor": question,
            "positive": f"{disease} {pos_category}: {pool[disease][pos_category]}",
            "negative": f"{disease} {neg_category}: {pool[disease][neg_category]}",
            "pos_category": pos_category,
            "neg_category": neg_category,
        })
        if (i + 1) % 10 == 0:
            print(f"  {len(synthetic)}/{N_SAMPLES} 생성 완료...")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(synthetic, f, ensure_ascii=False, indent=2)
    print(f"{len(synthetic)}건을 {out_path}에 저장했습니다.")


if __name__ == "__main__":
    main_run()
