"""처음부터 끝까지: 실제 vLLM(최종 모델) + FastAPI에 테스트 120건을 차트 + 대화로 보내고, 응답을 그대로 채점한다.

가짜 모델 서버로 나눠 테스트하던 것을 한 번에 실제로 붙여 확인한다.
  1. run      : 120건을 API(/v1/notes)로 보낸다. 응답의 보정한 기록·보정 전 출력을 ROUGE와 근거 검사로 채점하고,
                오프라인 평가(01_baseline + eval_correction)와 같은지 비교한다. 지연·재시도·보정 건수도 센다.
  2. stream   : /v1/notes/stream으로 3건 — 첫 토큰까지 시간, 끝까지 시간, done의 보정본이 일반 응답과 같은지
  3. overload : 동시 처리 상한보다 많은 요청을 한꺼번에 보내 대기·503(Retry-After)이 실제로 나는지
                (API를 QUEUE_TIMEOUT_S를 짧게 다시 띄운 뒤 실행)
  4. show     : 포트폴리오 스크린샷용. 실제 요청 하나의 입력·응답·보정 내역과 120건 요약을 보기 좋게 출력

    python kaggle/11_e2e.py --phase run stream --concurrency 8
    python kaggle/11_e2e.py --phase overload --burst 24
출력: outputs/e2e/<tag>_api.jsonl, results/e2e_<tag>.json
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

try:
    import httpx2 as httpx  # openai 3.x 환경
except ImportError:
    import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import evaluate, load_jsonl, parse_prediction  # noqa: E402
from faithfulness import TEST_SPLITS, check, summarize  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


async def post_note(client, url, case, chart):
    start = time.perf_counter()
    r = await client.post(f"{url}/v1/notes", json={"dialogue": case["dialogue"], "chart": chart})
    out = {"id": case["id"], "status": r.status_code, "client_s": round(time.perf_counter() - start, 2)}
    if r.status_code == 200:
        out |= r.json()
    else:
        out |= {"detail": r.text[:200], "retry_after": r.headers.get("retry-after")}
    return out


async def phase_run(args, cases, charts, report):
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=600) as client:
        health = (await client.get(f"{args.url}/health")).json()

        async def go(c):
            async with sem:
                return await post_note(client, args.url, c, charts[c["id"]]["chart"])
        start = time.perf_counter()
        rows = await asyncio.gather(*[go(c) for c in cases])
        wall = time.perf_counter() - start

    out_dir = ROOT / "outputs" / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.tag}_api.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r["status"] == 200]
    final = {r["id"]: json.dumps(r["note"]) for r in ok}
    model = {r["id"]: json.dumps(r["model_note"]) for r in ok}
    scored = [c for c in cases if c["id"] in final]

    def faith(p):
        s = summarize([check(parse_prediction(p[c["id"]])[0], charts[c["id"]]["chart_text"] + "\n\n" + c["dialogue"])
                       for c in scored])
        return s | {"supported_numbers": s["numbers_total"] - s["unsupported_numbers"]}

    rouge_final, _ = evaluate(scored, final)
    rouge_model, _ = evaluate(scored, model)
    lat = [r["latency_s"] for r in ok]
    report["run"] = {
        "health": health, "requests": len(rows), "ok": len(ok),
        "status": {str(s): sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "concurrency": args.concurrency, "wall_time_s": round(wall, 1),
        "latency_s": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "max": max(lat) if lat else None},
        "retried": sum(r["attempts"] > 1 for r in ok),
        "completion_tokens_mean": round(statistics.mean(r["completion_tokens"] for r in ok), 1) if ok else None,
        "corrections": sum(len(r["corrections"]) for r in ok),
        "needs_review": sum(r["review"]["needs_review"] for r in ok),
        "rouge1_model": rouge_model["ALL"]["rouge1"], "rouge1_final": rouge_final["ALL"]["rouge1"],
        "faithfulness_model": faith(model), "faithfulness_final": faith(final),
    }
    # 오프라인 평가와 같은가: 같은 모델·입력·설정이면 보정 전 출력이 오프라인 생성과 같아야 한다
    if args.offline == "none":  # 다시 학습한 모델이면 비교할 오프라인 생성이 없다
        return {r["id"]: r for r in ok}
    offline = {p["id"]: p["prediction"] for p in load_jsonl(ROOT / args.offline)}
    strip = lambda d: {k: str(v).strip() for k, v in (d or {}).items()}  # noqa: E731 (끝 공백 차이는 같은 출력으로 본다)
    same = sum(strip(parse_prediction(offline[r["id"]])[1]) == strip(r["model_note"]) for r in ok if r["id"] in offline)
    report["run"]["same_as_offline_generation"] = f"{same}/{len(ok)}"
    return {r["id"]: r for r in ok}


async def phase_stream(args, cases, charts, report, run_rows):
    rows = []
    async with httpx.AsyncClient(timeout=600) as client:
        for c in cases[:3]:
            start, first, done = time.perf_counter(), None, None
            async with client.stream("POST", f"{args.url}/v1/notes/stream",
                                     json={"dialogue": c["dialogue"], "chart": charts[c["id"]]["chart"]}) as r:
                buf = ""
                async for chunk in r.aiter_text():
                    buf += chunk
                    if first is None and "event: token" in buf:
                        first = time.perf_counter() - start
                    if "event: done" in buf or "event: error" in buf:
                        done = time.perf_counter() - start
            last = buf.strip().split("\n\n")[-1]
            name = last.split("\n")[0].removeprefix("event: ")
            data = json.loads(last.split("\n")[1].removeprefix("data: "))
            rows.append({"id": c["id"], "event": name, "first_token_s": round(first or -1, 2),
                         "done_s": round(done or -1, 2), "corrections": len(data.get("corrections", [])),
                         "same_note_as_non_stream": c["id"] in run_rows and data.get("note") == run_rows[c["id"]]["note"]})
    report["stream"] = rows


async def phase_overload(args, cases, charts, report):
    async with httpx.AsyncClient(timeout=600) as client:
        start = time.perf_counter()
        rows = await asyncio.gather(*[post_note(client, args.url, c, charts[c["id"]]["chart"])
                                      for c in cases[:args.burst]])
        wall = time.perf_counter() - start
        metrics = (await client.get(f"{args.url}/metrics")).json()
    report["overload"] = {
        "burst": args.burst, "wall_time_s": round(wall, 1),
        "status": {str(s): sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "retry_after": sorted({r.get("retry_after") for r in rows if r["status"] == 503}),
        "ok_client_s": {"p50": pct([r["client_s"] for r in rows if r["status"] == 200], 0.5),
                        "max": max((r["client_s"] for r in rows if r["status"] == 200), default=None)},
        "rejected_client_s": pct([r["client_s"] for r in rows if r["status"] == 503], 0.5),
        "server_metrics": metrics,
    }


async def phase_show(args, cases, charts, report):
    """포트폴리오 스크린샷용: 실제 요청 하나를 보내 입력(차트·대화 일부)과 응답(보정한 기록·뺀 값·확인 표시)을 보기 좋게 출력한다."""
    case = next(c for c in cases if c["id"] == args.show_id)
    chart = charts[case["id"]]["chart"]
    async with httpx.AsyncClient(timeout=600) as client:
        r = await post_note(client, args.url, case, chart)
    line = "─" * 96
    print(f"{line}\nPOST /v1/notes   ({case['id']}, 실제 vLLM + FastAPI)\n{line}")
    print("요청 chart:", json.dumps({k: v for k, v in chart.items() if v}, ensure_ascii=False, indent=1)[:900])
    print("요청 dialogue:", case["dialogue"][:300].replace("\n", " "), "...")
    print(f"{line}\n응답 {r['status']}  latency {r.get('latency_s')} s · attempts {r.get('attempts')} · "
          f"completion_tokens {r.get('completion_tokens')}\n{line}")
    for c in r.get("corrections", []):
        sec = r["model_note"][c["division"]]
        i = sec.find(c["removed"])
        before = sec[max(0, i - 90):i + len(c["removed"]) + 40].replace("\n", " ")
        print(f"[보정] {c['division']} · {c['kind']} \"{c['removed']}\" 뺌\n  모델 출력: ...{before}...")
        j = r["note"][c["division"]].find(before[:40].strip()[:25])
        print(f"  보정 결과: ...{r['note'][c['division']][max(0, j):j + 140].replace(chr(10), ' ')}...")
    print("review:", json.dumps(r.get("review"), ensure_ascii=False))
    print(f"{line}\n보정한 기록 (note_text 앞부분)\n{line}\n{r.get('note_text', '')[:1200]}")
    report["show"] = {k: r.get(k) for k in ("id", "status", "latency_s", "corrections", "review")}
    if "run" in report:
        s = report["run"]
        print(f"{line}\n테스트 120건 전체 (API 경유)\n{line}")
        print(f"성공 {s['ok']}/{s['requests']} · 동시 {s['concurrency']} · 전체 {s['wall_time_s']} s · "
              f"응답 p50 {s['latency_s']['p50']} s / p95 {s['latency_s']['p95']} s")
        print(f"ROUGE-1 보정 전 {s['rouge1_model']} → 보정 후 {s['rouge1_final']}")
        print(f"근거 없는 숫자 {s['faithfulness_model']['unsupported_number_rate']}% → "
              f"{s['faithfulness_final']['unsupported_number_rate']}% · 보정 {s['corrections']}건 · "
              f"오프라인 생성과 동일 {s.get('same_as_offline_generation', '-')}")


async def main_async(args):
    cases = [g for s in TEST_SPLITS for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]
    charts = {r["id"]: r for s in TEST_SPLITS for r in load_jsonl(ROOT / args.chart_dir / f"{s}.jsonl")}
    path = ROOT / "results" / f"e2e_{args.tag}.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    run_rows = {}
    if "run" in args.phase:
        run_rows = await phase_run(args, cases, charts, report)
    if "stream" in args.phase:
        await phase_stream(args, cases, charts, report, run_rows)
    if "overload" in args.phase:
        await phase_overload(args, cases, charts, report)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if "show" in args.phase:
        await phase_show(args, cases, charts, report)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--phase", nargs="+", default=["run", "stream"], choices=["run", "stream", "overload", "show"])
    ap.add_argument("--concurrency", type=int, default=8, help="T4×2 측정에서 30초 안에 받는 동시 요청 수")
    ap.add_argument("--burst", type=int, default=24)
    ap.add_argument("--show-id", default="D2N164", help="show 단계에서 보낼 사례 (보정이 일어나는 예)")
    ap.add_argument("--tag", default="qwen3b-r16-chart4")
    ap.add_argument("--chart-dir", default="data/chart")
    ap.add_argument("--offline", default="outputs/sft/qwen3b-r16-chart4-fp0.4_json.jsonl", help="none: 비교하지 않음")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
