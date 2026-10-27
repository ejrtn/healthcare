"""DPO 1단계: 학습용 대화마다 SFT 모델로 진료기록 후보를 여러 개 만든다.

같은 대화라도 샘플링하면 어떤 후보는 대화대로 쓰고 어떤 후보는 숫자를 지어내거나 대화와 반대로 쓴다.
이 후보들을 채점해(06_judge.py + src/faithfulness.py) 좋은 답·나쁜 답 짝을 고른다(src/build_dpo_pairs.py).
모델 자신의 출력으로 짝을 만들어야(on-policy) "이 모델이 실제로 하는 실수"를 고친다.

    !python kaggle/05_dpo_candidates.py --model outputs/sft/qwen3b-r16-grounded/merged

결과: outputs/dpo/candidates_{train,valid}.jsonl — {"id", "sample", "prediction", "finish_reason"}
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle"))

from evaluate import load_jsonl  # noqa: E402

json_sampling_params = __import__("01_baseline").json_sampling_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="data/sft_grounded")
    ap.add_argument("--n", type=int, default=8, help="대화당 후보 수")
    # 너무 낮으면 후보가 다 비슷해 짝을 못 만들고, 너무 높으면 문장이 망가진 후보만 늘어난다
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--frequency-penalty", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="float16", max_model_len=8192, gpu_memory_utilization=0.90, seed=0)
    params = json_sampling_params(
        SamplingParams, n=args.n, temperature=args.temperature, top_p=0.95, seed=0,
        frequency_penalty=args.frequency_penalty, max_tokens=args.max_tokens,
    )
    out_dir = ROOT / "outputs" / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid"]:
        rows = load_jsonl(ROOT / args.data_dir / f"{split}.jsonl")
        outputs = llm.chat([r["messages"][:-1] for r in rows], params)
        n_trunc = 0
        with open(out_dir / f"candidates_{split}.jsonl", "w", encoding="utf-8") as f:
            for r, o in zip(rows, outputs):
                for k, c in enumerate(o.outputs):
                    n_trunc += c.finish_reason == "length"
                    f.write(json.dumps({"id": r["id"], "sample": k, "prediction": c.text,
                                        "finish_reason": c.finish_reason}, ensure_ascii=False) + "\n")
        print(f"{split}: 대화 {len(rows)}건 × 후보 {args.n}개, 잘림 {n_trunc}개")


if __name__ == "__main__":
    main()
