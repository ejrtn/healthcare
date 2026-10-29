"""근거 확인 에이전트를 테스트 120건 출력에 적용하고, 전후를 같은 지표로 잰다 (src/agent.py).

vLLM을 도구 호출 옵션으로 먼저 띄운다 (Qwen2.5는 <tool_call> 형식 → hermes 파서가 OpenAI tool_calls로 바꾼다):
    vllm serve Qwen/Qwen2.5-3B-Instruct --served-model-name agent --dtype float16 --max-model-len 8192 \
        --enable-auto-tool-choice --tool-call-parser hermes
그다음:
    python kaggle/10_agent.py --preds outputs/sft/qwen3b-r16-chart4-fp0.4_json.jsonl --tag agent-qwen3b

출력
  outputs/agent/<tag>_json.jsonl   고친 기록 (01_baseline 출력과 같은 형식 → evaluate·faithfulness로 그대로 채점)
  outputs/agent/<tag>_trace.jsonl  문장마다 도구 호출·판단·반영 여부
  results/agent_<tag>.json         ROUGE·근거 검사 전후, 판단 분포, 도구 호출 통계
"""
import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import TOOLS, apply_divisions, flagged_sentences, verify_sentence  # noqa: E402
from evaluate import evaluate, load_jsonl, parse_prediction  # noqa: E402
from evidence import split_turns  # noqa: E402
from faithfulness import TEST_SPLITS, check, summarize  # noqa: E402


def label_accuracy(preds_path, cases, outs):
    """data/agent_labels.json(사람이 대조한 정답)으로 판단을 채점한다.
    tp(실제로 근거 없음)는 고치거나 지워야 맞고, fp(검사기 오탐)는 남겨야 맞다. "전부 삭제" 기준선은 tp 100%, fp 0%다."""
    path = ROOT / "data" / "agent_labels.json"
    if not path.exists():
        return None
    labels = [x for x in json.loads(path.read_text(encoding="utf-8"))["items"] if x["preds"] == preds_path]
    if not labels:
        return None
    by_id = {c["id"]: rs for c, (_, rs) in zip(cases, outs)}
    rows = []
    for x in labels:
        hit = [r for r in by_id.get(x["id"], []) if r["sentence"].startswith(x["sentence_start"])]
        changed = bool(hit) and hit[0]["applied"]
        rows.append({"id": x["id"], "label": x["label"], "verdict": hit[0]["verdict"] if hit else None,
                     "correct": changed if x["label"] == "tp" else not changed})
    tp = [r for r in rows if r["label"] == "tp"]
    fp = [r for r in rows if r["label"] == "fp"]
    return {"sentences": len(rows), "accuracy": round(100 * sum(r["correct"] for r in rows) / len(rows), 1),
            "tp_changed": f"{sum(r['correct'] for r in tp)}/{len(tp)}",
            "fp_kept": f"{sum(r['correct'] for r in fp)}/{len(fp)}",
            "remove_all_accuracy": round(100 * len(tp) / len(rows), 1), "rows": rows}


async def run_case(client, model, case, pred, chart, chart_text, method, sem):
    note, divisions = parse_prediction(pred)
    if divisions is None:  # JSON이 깨진 출력은 그대로 둔다
        return pred, []
    source = chart_text + "\n\n" + case["dialogue"]
    items = flagged_sentences(note, check(note, source))
    turns = split_turns(case["dialogue"])
    async with sem:
        results = [await verify_sentence(client, model, it, turns, chart, source, method) for it in items]
    return json.dumps(apply_divisions(divisions, results), ensure_ascii=False), results


async def main_async(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.base_url, api_key="unused", timeout=600, max_retries=0)
    cases = [g for s in TEST_SPLITS for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]
    charts = {r["id"]: r for s in TEST_SPLITS for r in load_jsonl(ROOT / args.chart_dir / f"{s}.jsonl")}
    preds = {p["id"]: p["prediction"] for p in load_jsonl(ROOT / args.preds)}
    sem = asyncio.Semaphore(args.concurrency)

    start = time.perf_counter()
    outs = await asyncio.gather(*[
        run_case(client, args.model, c, preds[c["id"]], charts[c["id"]]["chart"], charts[c["id"]]["chart_text"],
                 args.method, sem) for c in cases])
    elapsed = time.perf_counter() - start

    out_dir = ROOT / "outputs" / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    revised = {c["id"]: o[0] for c, o in zip(cases, outs)}
    with open(out_dir / f"{args.tag}_json.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps({"id": c["id"], "prediction": revised[c["id"]]}, ensure_ascii=False) + "\n")
    with open(out_dir / f"{args.tag}_trace.jsonl", "w", encoding="utf-8") as f:
        for c, (_, results) in zip(cases, outs):
            f.write(json.dumps({"id": c["id"], "sentences": results}, ensure_ascii=False) + "\n")

    def faith(p):
        rows = []
        for c in cases:
            note, _ = parse_prediction(p[c["id"]])
            rows.append(check(note, charts[c["id"]]["chart_text"] + "\n\n" + c["dialogue"]))
        s = summarize(rows)
        return s | {"supported_numbers": s["numbers_total"] - s["unsupported_numbers"]}  # 맞는 숫자를 지우지 않았나

    # 기준선: LLM 없이 표시된 문장을 전부 지운다. 에이전트는 이보다 맞는 정보를 더 살려야 의미가 있다
    remove_all = {}
    for c, (_, rs) in zip(cases, outs):
        _, divisions = parse_prediction(preds[c["id"]])
        remove_all[c["id"]] = preds[c["id"]] if divisions is None else json.dumps(
            apply_divisions(divisions, [r | {"verdict": "remove", "applied": True} for r in rs]), ensure_ascii=False)

    report_labels = label_accuracy(args.preds, cases, outs)
    results = [r for _, rs in outs for r in rs]
    calls = [t for r in results for t in r["tool_calls"]]
    tool_names = {t["function"]["name"] for t in TOOLS}
    rouge_before, _ = evaluate(cases, preds)
    rouge_after, _ = evaluate(cases, revised)
    rouge_remove_all, _ = evaluate(cases, remove_all)
    report = {
        "model": args.model, "method": args.method, "preds": args.preds, "wall_time_s": round(elapsed, 1),
        "sentences": len(results),
        "verdicts": dict(Counter(r["verdict"] for r in results)),
        "applied": dict(Counter(r["verdict"] for r in results if r["applied"])),
        "not_applied_reasons": dict(Counter(r["reason"] for r in results if not r["applied"] and r["reason"])),
        "tool_calls": {
            "total": len(calls),
            "per_sentence": round(len(calls) / max(1, len(results)), 2),
            "sentences_without_call": sum(not r["tool_calls"] for r in results),
            "valid_name": sum(t["tool"] in tool_names for t in calls),
            "by_tool": dict(Counter(t["tool"] for t in calls)),
        },
        "rouge1_before": rouge_before["ALL"]["rouge1"], "rouge1_after": rouge_after["ALL"]["rouge1"],
        "rouge1_remove_all": rouge_remove_all["ALL"]["rouge1"],
        "rouge_after": rouge_after,
        "faithfulness_before": faith(preds), "faithfulness_after": faith(revised),
        "faithfulness_remove_all": faith(remove_all),
        "labels": report_labels,
    }
    (ROOT / "results" / f"agent_{args.tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rouge_after"}, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="agent", help="vllm serve --served-model-name")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--chart-dir", default="data/chart")
    ap.add_argument("--method", default="bm25", choices=["bm25", "embedding", "hybrid"])
    ap.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
