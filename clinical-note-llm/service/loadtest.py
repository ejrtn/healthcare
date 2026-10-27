"""API 부하 테스트: 동시 사용자 수를 늘려 가며 /v1/notes를 호출한다.

03_serving_bench.py가 모델 서버 자체를 잰다면, 이것은 API 계층까지 포함한 실제 서비스 경로를 잰다
(JSON 검증·재시도·동시 처리 상한·대기열 거절 포함). 입력은 ACI-BENCH 테스트 대화를 차례로 쓴다.

    python service/loadtest.py --url http://127.0.0.1:8080 --users 1 2 4 --requests 8
결과: results/loadtest_<tag>.json
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import httpx2 as httpx  # openai 3.x 환경
except ImportError:
    import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import load_jsonl  # noqa: E402


async def one(client, url, dialogue):
    start = time.perf_counter()
    try:
        r = await client.post(f"{url}/v1/notes", json={"dialogue": dialogue})
        status = r.status_code
        body = r.json() if status == 200 else {}
    except httpx.HTTPError as e:
        status, body = type(e).__name__, {}
    return {"status": status, "latency": time.perf_counter() - start,
            "attempts": body.get("attempts"), "completion_tokens": body.get("completion_tokens", 0)}


async def run_level(client, url, dialogues, users, n, offset):
    sem = asyncio.Semaphore(users)

    async def bounded(i):
        async with sem:
            return await one(client, url, dialogues[(offset + i) % len(dialogues)])

    start = time.perf_counter()
    results = await asyncio.gather(*(bounded(i) for i in range(n)))
    wall = time.perf_counter() - start
    ok = [r for r in results if r["status"] == 200]
    lat = sorted(r["latency"] for r in ok)
    return {
        "users": users, "requests": n,
        "status": dict(Counter(str(r["status"]) for r in results)),
        "retried": sum(r["attempts"] == 2 for r in ok),
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2) if lat else None,
        "req_per_min": round(60 * len(ok) / wall, 2),
        "output_tok_per_s": round(sum(r["completion_tokens"] for r in ok) / wall, 1),
    }


async def main_async(args):
    dialogues = [g["dialogue"] for s in ["test1", "test2", "test3"]
                 for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        print((await client.get(f"{args.url}/health")).json())
        levels, offset = [], 0
        for users in args.users:
            n = max(args.requests, 2 * users)
            level = await run_level(client, args.url, dialogues, users, n, offset)
            offset += n
            print(json.dumps(level, ensure_ascii=False))
            levels.append(level)
        metrics = (await client.get(f"{args.url}/metrics")).json()
    return levels, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--users", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--requests", type=int, default=8, help="단계별 최소 요청 수")
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()

    levels, metrics = asyncio.run(main_async(args))
    out = ROOT / "results" / f"loadtest_{args.tag}.json"
    out.write_text(json.dumps({"url": args.url, "levels": levels, "server_metrics": metrics},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
