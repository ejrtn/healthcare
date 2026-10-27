"""OpenAI 호환 서버(vLLM, llama.cpp 등)에 테스트 대화를 보내 품질을 채점한다.

01_baseline.py는 vLLM을 파이썬 안에서 직접 돌리지만, 이 스크립트는 서버 API만 쓰므로 어떤 모델 서버든
같은 기준으로 잴 수 있다 — CPU(llama.cpp + GGUF 4bit)로 바꿨을 때 품질이 얼마나 떨어지는지 보는 용도.

CPU는 느리므로 --limit으로 일부만 돌리고, 비교 대상(GPU FP16 예측)도 같은 사건만 골라 채점한다.

    python src/eval_server.py --base-url http://127.0.0.1:8000/v1 --tag q4km-cpu --limit 20 \
        --compare outputs/sft/qwen3b-r16-fp0.4_json.jsonl
결과: outputs/server/<tag>_json.jsonl, results/server_<tag>_json.json
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from evaluate import DIVISIONS, evaluate, load_jsonl
from prompts import build_messages
from schema import NOTE_JSON_SCHEMA

ROOT = Path(__file__).resolve().parent.parent
TEST_SPLITS = ["test1", "test2", "test3"]


async def generate(client, args, case):
    start = time.perf_counter()
    resp = await client.chat.completions.create(
        model=args.model, messages=build_messages(case["dialogue"], json_mode=True),
        temperature=0, max_tokens=args.max_tokens, frequency_penalty=args.frequency_penalty,
        response_format={"type": "json_schema", "json_schema": {"name": "clinical_note", "schema": NOTE_JSON_SCHEMA}},
    )
    choice = resp.choices[0]
    return {
        "id": case["id"], "prediction": choice.message.content or "",
        "finish_reason": choice.finish_reason, "latency_s": round(time.perf_counter() - start, 2),
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }


def pick_cases(limit, seed):
    cases = [g for s in TEST_SPLITS for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]
    if limit and limit < len(cases):
        # 출처(aci/virtassist/virtscribe) 비율을 유지하도록 출처별로 고르게 뽑는다
        rng = random.Random(seed)
        by_src = {}
        for c in cases:
            by_src.setdefault(c["dataset"], []).append(c)
        picked = []
        for src, group in sorted(by_src.items()):
            k = max(1, round(limit * len(group) / len(cases)))
            picked += rng.sample(group, k)
        cases = sorted(picked[:limit], key=lambda c: c["id"])
    return cases


async def main_async(args, cases):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.base_url, api_key="unused", timeout=args.timeout, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    out_path = ROOT / "outputs" / "server" / f"{args.tag}_json.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in load_jsonl(out_path)} if out_path.exists() else {}  # 중단 후 이어 돌리기

    async def bounded(case):
        async with sem:
            r = await generate(client, args, case)
            print(f"{r['id']}: {r['latency_s']}s, {r['completion_tokens']} tok, {r['finish_reason']}", flush=True)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return r

    start = time.perf_counter()
    new = await asyncio.gather(*(bounded(c) for c in cases if c["id"] not in done))
    wall = time.perf_counter() - start
    await client.close()
    for r in new:
        done[r["id"]] = r
    return done, new, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="note-model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0이면 120건 전체")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--frequency-penalty", type=float, default=0.4)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--compare", nargs="*", default=[], help="같은 사건으로 채점할 다른 예측 파일")
    args = ap.parse_args()

    cases = pick_cases(args.limit, args.seed)
    print(f"{len(cases)}건: {[c['id'] for c in cases]}")
    results, new, wall = asyncio.run(main_async(args, cases))

    preds = {cid: r["prediction"] for cid, r in results.items()}
    summary, per_case = evaluate(cases, preds)
    rows = [results[c["id"]] for c in cases]
    summary["generation"] = {
        "base_url": args.base_url, "frequency_penalty": args.frequency_penalty,
        "num_requests": len(cases), "concurrency": args.concurrency,
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 1),
        "output_tok_per_s": round(sum(r["completion_tokens"] for r in new) / wall, 1) if new else None,
    }
    comparisons = {}
    for path in args.compare:
        other = {p["id"]: p["prediction"] for p in load_jsonl(ROOT / path)}
        comparisons[path], _ = evaluate(cases, other)
    report = {"summary": summary, "same_cases": comparisons, "case_ids": [c["id"] for c in cases], "per_case": per_case}
    (ROOT / "results" / f"server_{args.tag}_json.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def line(name, s):
        return f"| {name} | {s['ALL']['rouge1']:.1f} | " + " | ".join(f"{s[d]['rouge1']:.1f}" for d in DIVISIONS) + \
               f" | {100 * s['json_valid']:.0f}% |"
    print("\n| 시스템 | ROUGE-1 | S | O-exam | O-results | A&P | JSON 유효 |\n|---|---|---|---|---|---|---|")
    print(line(args.tag, summary))
    for path, s in comparisons.items():
        print(line(Path(path).stem, s))
    print(json.dumps(summary["generation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
