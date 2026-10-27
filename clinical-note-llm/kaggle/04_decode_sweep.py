"""반복 억제 설정 탐색 — 검증셋(ACI valid 20건)에서만 고른다.

SFT 3B는 테스트 120건 중 34건이 같은 줄을 반복하다 max_tokens에 걸렸다
(예: "• Endocrine: Denies weight changes."가 수십 번). 반복 억제 벌점으로 이를 줄이되,
테스트셋을 보고 값을 고르면 테스트 점수가 부풀려지므로 검증셋에서 고른 값을 테스트에 한 번만 쓴다.

선택 기준: 4구역 ROUGE-1 평균 (잘린 출력은 JSON이 깨져 점수가 크게 깎이므로 반복 억제 효과가 반영된다)

Kaggle:
    %cd /kaggle/working/healthcare/clinical-note-llm
    !python kaggle/04_decode_sweep.py --model outputs/sft/qwen3b-r16/merged
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIVISIONS = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]

# (repetition_penalty, frequency_penalty)
CONFIGS = [(1.0, 0.0), (1.0, 0.1), (1.0, 0.2), (1.0, 0.4), (1.05, 0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="sft3b", help="결과 파일 이름 앞부분")
    args = ap.parse_args()

    rows = []
    for rp, fp in CONFIGS:
        tag = f"{args.name}-rp{rp}-fp{fp}"
        subprocess.run([
            sys.executable, str(ROOT / "kaggle" / "01_baseline.py"),
            "--model", args.model, "--mode", "json", "--split", "valid", "--group", "sweep",
            "--tag", tag, "--repetition-penalty", str(rp), "--frequency-penalty", str(fp),
        ], check=True)
        s = json.loads((ROOT / "results" / f"sweep_{tag}_json.json").read_text(encoding="utf-8"))["summary"]
        div_mean = sum(s[d]["rouge1"] for d in DIVISIONS) / len(DIVISIONS)
        rows.append((rp, fp, div_mean, s))

    print("\n| rep_penalty | freq_penalty | 구역 평균 R1 | 전체 R1 | JSON 유효 | 잘림 |")
    print("|---|---|---|---|---|---|")
    for rp, fp, div_mean, s in rows:
        print(f"| {rp} | {fp} | {div_mean:.1f} | {s['ALL']['rouge1']} | "
              f"{100 * s['json_valid']:.0f}% | {s['generation']['truncated']} |")
    best = max(rows, key=lambda r: r[2])
    print(f"\n선택: repetition_penalty={best[0]}, frequency_penalty={best[1]} (구역 평균 R1 {best[2]:.1f})")


if __name__ == "__main__":
    main()
