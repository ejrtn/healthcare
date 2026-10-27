"""실험 결과를 한 표로 모은다: results/runs.md

- 우리 실험: outputs/<group>/<tag>.jsonl 예측을 엄격/관대(--lenient) 두 방식으로 재채점하고,
  results/<group>_<tag>.json의 생성 통계(잘림 수, 처리량)를 붙인다.
- 비교 기준: results/baselines.json (GPT-4 등 기존 모델 재채점 결과)
"""
import json
from pathlib import Path

from evaluate import DIVISIONS, evaluate, load_jsonl

ROOT = Path(__file__).resolve().parent.parent
TEST_SPLITS = ["test1", "test2", "test3"]
GROUPS = ["baseline", "sft", "dpo"]
REFERENCE_SYSTEMS = ["GPT-4 (zero-shot)", "ChatGPT (zero-shot)", "BART+SAMSum FT (division)"]


def row(name, s, json_valid="-", truncated="-", tok_s="-"):
    divs = " | ".join(f"{s[d]['rouge1']:.1f}" for d in DIVISIONS)
    return f"| {name} | {s['ALL']['rouge1']:.1f} | {s['ALL']['rougeL']:.1f} | {divs} | {json_valid} | {truncated} | {tok_s} |"


def main():
    golds = [g for t in TEST_SPLITS for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{t}.jsonl")]
    lines = [
        "| 시스템 | ROUGE-1 | ROUGE-L | S | O-exam | O-results | A&P | JSON 유효 | 잘림 | 출력 tok/s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    reference = json.loads((ROOT / "results" / "baselines.json").read_text(encoding="utf-8"))
    for name in REFERENCE_SYSTEMS:
        lines.append(row(f"{name} †", reference[name]))

    for group in GROUPS:
        for pred_path in sorted((ROOT / "outputs" / group).glob("*.jsonl")):
            if pred_path.name.startswith("candidates_"):  # DPO 학습용 후보 (학습 대화라 정답 채점 대상이 아니다)
                continue
            tag = pred_path.stem
            preds = {p["id"]: p["prediction"] for p in load_jsonl(pred_path)}
            gen = {}
            stats_path = ROOT / "results" / f"{group}_{tag}.json"
            if stats_path.exists():
                gen = json.loads(stats_path.read_text(encoding="utf-8"))["summary"].get("generation", {})

            strict, _ = evaluate(golds, preds)
            json_valid = f"{100 * strict['json_valid']:.1f}%" if tag.endswith("_json") else "-"
            extra = dict(json_valid=json_valid, truncated=gen.get("truncated", "-"),
                         tok_s=gen.get("output_tok_per_s", "-"))
            lines.append(row(f"{group}: {tag}", strict, **extra))
            if tag.endswith("_text"):
                lenient, _ = evaluate(golds, preds, lenient=True)
                lines.append(row(f"{group}: {tag} (마크다운 제거) ‡", lenient, **extra))

    table = "\n".join(lines)
    note = (
        "ACI-BENCH test1~3, 120건. 구역 열은 구역별 ROUGE-1.\n"
        "† 저장소에 공개된 예측을 같은 채점기로 재채점.\n"
        "‡ 참고용: 모델이 '**제목:**' 마크다운으로 써서 구역 분리기가 제목을 못 알아본 것을 보정한 점수.\n"
    )
    (ROOT / "results" / "runs.md").write_text(note + "\n" + table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
