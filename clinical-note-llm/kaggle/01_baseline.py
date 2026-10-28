"""기준선: 학습 안 한 오픈 모델에 프롬프트만으로 진료기록을 생성하고 채점한다.

비교 축
  - 모델 크기: 3B (T4 1장) vs 7B (T4 2장, tensor parallel)
  - 출력 방식: 자유 텍스트(text) vs JSON 스키마 강제(json, vLLM structured output)

Kaggle 실행 (GPU T4 x2, Internet On). vLLM 엔진이 GPU 메모리를 확실히 돌려주도록
실행마다 별도 프로세스로 띄운다:
    !pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio
    !git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git
    %cd healthcare/clinical-note-llm
    !python kaggle/01_baseline.py --model Qwen/Qwen2.5-3B-Instruct --mode text
    !python kaggle/01_baseline.py --model Qwen/Qwen2.5-3B-Instruct --mode json
    !python kaggle/01_baseline.py --model Qwen/Qwen2.5-7B-Instruct --mode json --tp 2

결과: outputs/<group>/<tag>_<mode>.jsonl (예측), results/<group>_<tag>_<mode>.json (점수·생성 통계)
최종 비교표는 로컬에서 src/summarize_runs.py로 다시 채점해 만든다 (results/runs.md).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import evaluate, load_jsonl  # noqa: E402
from prompts import build_messages  # noqa: E402
from schema import NOTE_JSON_SCHEMA  # noqa: E402

TEST_SPLITS = ["test1", "test2", "test3"]


def json_sampling_params(SamplingParams, **kwargs):
    # vLLM 버전에 따라 structured output 인터페이스 이름이 다르다
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(structured_outputs=StructuredOutputsParams(json=NOTE_JSON_SCHEMA), **kwargs)
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(guided_decoding=GuidedDecodingParams(json=NOTE_JSON_SCHEMA), **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["text", "json"], required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--tag", help="결과 파일 이름 (기본: 모델 이름). 로컬 경로 모델에 쓴다")
    ap.add_argument("--group", default="baseline", help="결과 묶음 (baseline / sft / sweep)")
    ap.add_argument("--split", choices=["test", "valid"], default="test",
                    help="valid: 생성 설정을 고르는 용도 (테스트셋을 보고 고르면 점수가 부풀려진다)")
    # 반복 억제. repetition_penalty는 입력(대화)에 나온 토큰까지 벌점을 줘서 대화의 의학 용어를
    # 옮겨 적는 데 방해가 될 수 있고, frequency_penalty는 생성한 토큰에만 반복 횟수만큼 벌점을 준다.
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--quantization", help="vLLM 양자화 방식 (양자화 후 품질 확인용)")
    ap.add_argument("--chart-dir", help="data/chart: 입력을 '차트 + 대화'로 준다 (src/build_chart_data.py)")
    ap.add_argument("--chart-field", default="chart_text",
                    help="chart_text: 나이·성별을 채운 차트(실제 EHR처럼) / chart_text_strict: 모르면 빈칸")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    splits = TEST_SPLITS if args.split == "test" else ["valid"]
    cases = [g for s in splits for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]

    llm = LLM(
        model=args.model,
        dtype="float16",  # T4에는 bfloat16 텐서코어가 없다
        tensor_parallel_size=args.tp,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        seed=0,
        quantization=args.quantization,
    )
    gen = dict(temperature=0, max_tokens=args.max_tokens,
               repetition_penalty=args.repetition_penalty, frequency_penalty=args.frequency_penalty)
    params = json_sampling_params(SamplingParams, **gen) if args.mode == "json" else SamplingParams(**gen)

    charts = {}
    if args.chart_dir:
        charts = {r["id"]: r[args.chart_field] for s in splits for r in load_jsonl(ROOT / args.chart_dir / f"{s}.jsonl")}
    conversations = [build_messages(c["dialogue"], json_mode=args.mode == "json", chart_text=charts.get(c["id"]))
                     for c in cases]
    start = time.perf_counter()
    outputs = llm.chat(conversations, params)
    elapsed = time.perf_counter() - start

    n_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    n_truncated = sum(o.outputs[0].finish_reason == "length" for o in outputs)
    preds = {c["id"]: o.outputs[0].text for c, o in zip(cases, outputs)}

    tag = f"{args.tag or args.model.split('/')[-1]}_{args.mode}"
    pred_dir = ROOT / "outputs" / args.group
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for cid, text in preds.items():
            f.write(json.dumps({"id": cid, "prediction": text}, ensure_ascii=False) + "\n")

    summary, per_case = evaluate(cases, preds)
    summary["generation"] = {
        "model": args.model, "quantization": args.quantization or "none", "mode": args.mode, "tensor_parallel": args.tp, "split": args.split,
        "repetition_penalty": args.repetition_penalty, "frequency_penalty": args.frequency_penalty,
        "chart": args.chart_field if args.chart_dir else None,
        "num_requests": len(cases), "wall_time_s": round(elapsed, 1),
        "output_tokens": n_out, "output_tok_per_s": round(n_out / elapsed, 1),
        "truncated": n_truncated,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"{args.group}_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_case": per_case}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
