"""ACI-BENCH 저장소에 들어 있는 기존 모델 예측(BART 계열 등)을
이 프로젝트 채점기(evaluate.py)로 다시 채점한다.

논문 표 숫자를 그대로 인용하지 않고 다시 채점하는 이유: 공식 결과 파일과
우리 채점 결과가 0.2~1점 정도 어긋났다(README 참고). 모든 시스템을 같은
채점기로 재야 우리 모델과의 비교가 공정하다.

출력: results/baselines.json, results/baselines.md (test1~3 합친 120건 기준)
"""
import csv
import json
from pathlib import Path
from statistics import mean

from evaluate import DIVISIONS, evaluate, load_jsonl

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "data" / "raw" / "aci-bench" / "baselines" / "predictions"
RESULTS = ROOT / "results"

TEST_FILES = {"test1": "clinicalnlp_taskB_test1", "test2": "clinicalnlp_taskC_test2", "test3": "clef_taskC_test3"}

# 모델 이름 → 테스트셋별 예측 파일명
SYSTEMS = {
    "BART+SAMSum FT (division)": lambda t, f: f"bart-large-xsum-samsum_{f}_division_combined.csv",
    "BART+SAMSum FT": lambda t, f: f"bart-large-xsum-samsum_{f}_full.csv",
    "BioBART FT": lambda t, f: f"BioBART_{f}_full.csv",
    "LED FT": lambda t, f: f"LED_{f}_full.csv",
    "Transcript copy": lambda t, f: f"transcript_{f}.csv",
}


def load_predictions(path, gold_ids):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["encounter_id"] for r in rows]
    assert ids == gold_ids, f"{path.name}: 사건 순서가 정답과 다름"
    return {r["encounter_id"]: str(r["note"]) for r in rows}


def main():
    RESULTS.mkdir(exist_ok=True)
    golds = {t: load_jsonl(ROOT / "data" / "processed" / f"aci_{t}.jsonl") for t in TEST_FILES}

    all_results = {}
    for name, fname_of in SYSTEMS.items():
        per_case = []
        for t, f in TEST_FILES.items():
            path = PRED_DIR / fname_of(t, f)
            if not path.exists():
                print(f"건너뜀: {name} ({path.name} 없음)")
                break
            preds = load_predictions(path, [g["id"] for g in golds[t]])
            per_case += evaluate(golds[t], preds)[1]
        else:
            all_results[name] = {
                key: {m: round(100 * mean(c[key][m] for c in per_case), 2) for m in ["rouge1", "rouge2", "rougeL"]}
                for key in ["ALL"] + DIVISIONS
            }
            print(f"{name}: ROUGE-1 {all_results[name]['ALL']['rouge1']}")

    with open(RESULTS / "baselines.json", "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)

    lines = [
        "| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | S | O-exam | O-results | A&P |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(all_results.items(), key=lambda kv: -kv[1]["ALL"]["rouge1"]):
        divs = " | ".join(str(r[d]["rouge1"]) for d in DIVISIONS)
        lines.append(f"| {name} | {r['ALL']['rouge1']} | {r['ALL']['rouge2']} | {r['ALL']['rougeL']} | {divs} |")
    table = "\n".join(lines)
    (RESULTS / "baselines.md").write_text(
        "ACI-BENCH test1~3 (120건), 이 프로젝트 채점기로 재채점. 구역 열은 구역별 ROUGE-1.\n\n" + table + "\n",
        encoding="utf-8",
    )
    print("\n" + table)


if __name__ == "__main__":
    main()
