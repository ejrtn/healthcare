"""vLLM 서빙 벤치마크: 실제 서버(OpenAI 호환 API)를 띄우고 동시 요청을 늘려 가며 측정한다.

측정 (동시 요청 수별)
  - TTFT: 요청 후 첫 토큰까지 (사용자가 체감하는 반응 속도)
  - TPOT: 첫 토큰 이후 토큰당 생성 시간 (스트리밍 속도)
  - E2E: 요청 하나의 전체 처리 시간 (진료기록 한 건 완성까지)
  - 처리량: 초당 완료 요청 수, 초당 출력 토큰 수
서버 로그에서 모델 메모리·KV 캐시 크기도 뽑는다 — 양자화로 줄어든 메모리가 KV 캐시(동시 처리
여력)로 얼마나 돌아가는지 보기 위해서다.

입력은 ACI-BENCH 테스트 대화(평균 약 1,700토큰)를 돌려 쓴다. 합성 프롬프트가 아니라 실제
서비스와 같은 길이 분포에서 재야 prefill 비용까지 현실적으로 잡힌다.

Kaggle (GPU T4 x2, vLLM 설치된 상태):
    %cd /kaggle/working/healthcare/clinical-note-llm
    !python kaggle/03_serving_bench.py --model outputs/sft/qwen3b-r16/merged --tag fp16-tp1
    !python kaggle/03_serving_bench.py --model outputs/sft/qwen3b-r16/merged --tag fp16-tp1-json --json
    !python kaggle/03_serving_bench.py --model outputs/sft/qwen3b-r16/merged --tag fp16-tp2 --tp 2
    !python kaggle/03_serving_bench.py --model Qwen/Qwen2.5-3B-Instruct-AWQ --tag awq4-base-tp1  # 4bit 속도 참고용 (공식 AWQ, 학습 전 가중치)

양자화 메모 (vLLM 0.32, T4): bitsandbytes는 지원 목록에서 빠졌고, int8_per_channel_weight_only는
일반(dense) 모델에서 "Quantizing 0 layers"로 아무 층도 양자화하지 않아 FP16과 같다.

결과: results/serving_<tag>.json, 서버 로그 outputs/serving/<tag>.log
"""
import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import load_jsonl  # noqa: E402
from prompts import build_messages  # noqa: E402
from schema import NOTE_JSON_SCHEMA  # noqa: E402

PORT = 8000
# "localhost"는 이름 해석(IPv6 시도 등) 지연이 TTFT에 섞일 수 있어 IP로 고정한다
BASE_URL = f"http://127.0.0.1:{PORT}"
SERVED_NAME = "note-model"

# 서버 로그에서 뽑을 줄 (vLLM 버전마다 문구가 조금씩 달라 넓게 잡는다)
LOG_PATTERNS = {
    "model_memory": re.compile(r"Model loading took ([\d.]+) ?GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "max_concurrency": re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
}


def start_server(args, log_path):
    cmd = [
        "vllm", "serve", args.model,
        "--served-model-name", SERVED_NAME,
        "--dtype", "float16",
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--port", str(PORT),
        "--seed", "0",
        # 모델의 generation_config.json(repetition_penalty 1.05 등)이 서버 기본값을 덮어쓰지 않게 한다 —
        # 품질 평가(오프라인 vLLM)와 같은 생성 설정으로 재야 한다
        "--generation-config", "vllm",
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"서버가 시작 중 종료됨 — 로그 확인: {log_path}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
                if r.status == 200:
                    return proc
        except OSError:
            pass
        time.sleep(5)
    proc.terminate()
    raise TimeoutError(f"{args.startup_timeout}초 안에 서버가 뜨지 않음 — 로그 확인: {log_path}")


def parse_server_log(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    found = {}
    for key, pattern in LOG_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[key] = float(matches[-1].replace(",", ""))
    # 문구가 바뀌어 정규식이 못 잡아도 원문은 남긴다
    keywords = ("KV cache", "GiB", "concurrency", "Quantiz", "attention backend")
    found["raw_lines"] = [line.strip() for line in text.splitlines()
                          if any(k in line for k in keywords)][-15:]
    return found


async def one_request(client, messages, args):
    extra = {}
    if args.json:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "clinical_note", "schema": NOTE_JSON_SCHEMA},
        }
    start = time.perf_counter()
    first = None
    n_tokens = 0
    stream = await client.chat.completions.create(
        model=SERVED_NAME, messages=messages, temperature=0, max_tokens=args.max_tokens,
        frequency_penalty=args.frequency_penalty,
        stream=True, stream_options={"include_usage": True}, **extra,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content and first is None:
            first = time.perf_counter()
        if chunk.usage:
            n_tokens = chunk.usage.completion_tokens
    end = time.perf_counter()
    first = first or end
    return {
        "ttft": first - start,
        "e2e": end - start,
        "tpot": (end - first) / max(1, n_tokens - 1),
        "output_tokens": n_tokens,
    }


async def run_level(client, prompts, concurrency, num_requests, offset, args):
    sem = asyncio.Semaphore(concurrency)

    async def bounded(i):
        async with sem:
            return await one_request(client, prompts[(offset + i) % len(prompts)], args)

    start = time.perf_counter()
    results = await asyncio.gather(*(bounded(i) for i in range(num_requests)))
    wall = time.perf_counter() - start

    def pct(values, q):
        values = sorted(values)
        return values[min(len(values) - 1, int(q * len(values)))]

    ttft = [r["ttft"] for r in results]
    e2e = [r["e2e"] for r in results]
    tpot = [r["tpot"] for r in results]
    total_tokens = sum(r["output_tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "ttft_p50_s": round(statistics.median(ttft), 3),
        "ttft_p95_s": round(pct(ttft, 0.95), 3),
        "tpot_p50_ms": round(1000 * statistics.median(tpot), 1),
        "e2e_p50_s": round(statistics.median(e2e), 2),
        "e2e_p95_s": round(pct(e2e, 0.95), 2),
        "req_per_s": round(num_requests / wall, 3),
        "output_tok_per_s": round(total_tokens / wall, 1),
        "mean_output_tokens": round(total_tokens / num_requests, 1),
    }


async def benchmark(args, prompts):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"{BASE_URL}/v1", api_key="unused", timeout=600)

    # 첫 요청은 컴파일·캐시 워밍업이 섞여 느리다 — 측정에서 뺀다
    await asyncio.gather(*(one_request(client, prompts[i], args) for i in range(2)))

    # 단계마다 이어지는 대화를 쓴다. 같은 대화를 다시 보내면 prefix cache가 prefill을 건너뛰어
    # TTFT·처리량이 실제보다 좋게 나온다 (공통 시스템 프롬프트 캐시는 실제 서비스에도 있으므로 유지).
    # 2 + 4 + 8 + 16 + 32 + 64 = 126건 중 마지막 6건만 대화 120건을 한 바퀴 돌아 재사용된다.
    offset = 2
    levels = []
    for c in args.concurrency:
        # 동시 요청 수의 2배(최소 4건)를 보내야 큐가 한동안 꽉 찬 상태가 유지된다
        n = max(4, 2 * c)
        level = await run_level(client, prompts, c, n, offset, args)
        offset += n
        print(json.dumps(level, ensure_ascii=False))
        levels.append(level)
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--quantization", help="vLLM --quantization 값 (예: int8_per_channel_weight_only, awq). vLLM 0.32에는 bitsandbytes가 없다")
    ap.add_argument("--json", action="store_true", help="JSON 스키마 강제(structured output)")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--max-tokens", type=int, default=1024)
    # 품질 평가에서 고른 반복 억제 값 — 서비스와 같은 설정으로 잰다
    ap.add_argument("--frequency-penalty", type=float, default=0.4)
    ap.add_argument("--startup-timeout", type=int, default=900)
    args = ap.parse_args()

    cases = [g for s in ["test1", "test2", "test3"]
             for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]
    prompts = [build_messages(c["dialogue"], json_mode=args.json) for c in cases]

    log_dir = ROOT / "outputs" / "serving"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.tag}.log"

    proc = start_server(args, log_path)
    try:
        levels = asyncio.run(benchmark(args, prompts))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    result = {
        "config": {"model": args.model, "tag": args.tag, "tp": args.tp,
                   "quantization": args.quantization or "none", "json": args.json,
                   "max_tokens": args.max_tokens, "frequency_penalty": args.frequency_penalty},
        "server": parse_server_log(log_path),
        "levels": levels,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"serving_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result["server"], ensure_ascii=False))


if __name__ == "__main__":
    main()
