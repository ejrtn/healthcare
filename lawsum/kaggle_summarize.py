"""Kaggle GPU(T4 x2)에서 vLLM으로 판례 요약을 돌린다 — 모델만 바꿔 비교하기 위한 실행기.

로컬(summarize.py)은 CPU에서 Llama-3.1-8B GGUF(4bit)를 쓴다. 한국어 특화 모델과 비교하려면 모델마다 CPU로
62건을 돌려야 해서, 같은 프롬프트(12차: 법리만 → 거의 비면 법리+적용)와 같은 후처리(섹션 제목·외톨이 번호 제거)를
그대로 가져와 실행 방식만 vLLM으로 바꿨다. 공정한 비교를 위해 Llama-3.1-8B도 같은 방식(FP16, GPU)으로 다시 돌린다.

    !python kaggle_summarize.py --model kakaocorp/kanana-1.5-8b-instruct-2505 --tag kanana-1.5-8b

결과: data/summarized_<tag>.json (summarize.py 출력과 같은 형식) → 로컬에서
    python evaluate.py --summarized data/summarized_<tag>.json --out data/evaluated_<tag>.json
"""
import argparse
import json
import time

from summarize import (
    SEMANTIC_PROMPT,
    SEMANTIC_PROMPT_WITH_APPLICATION,
    _EMPTY_RESULT_THRESHOLD,
    _strip_lone_leading_number,
    _strip_section_headers,
    strip_html,
)


def postprocess(text):
    return _strip_lone_leading_number(_strip_section_headers(text.strip()))


def run(llm, params, prompt, texts):
    outputs = llm.chat([[{"role": "system", "content": prompt}, {"role": "user", "content": t}] for t in texts],
                       params)
    return [postprocess(o.outputs[0].text) for o in outputs], outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tp", type=int, default=2, help="8B FP16(16GB)는 T4 한 장(15GB)에 안 들어가 2장에 나눈다")
    # summarize.py와 같은 0.2가 기본. A.X는 숫자를 한 자리씩 토큰으로 나눠, 0.2로 샘플링하면 연도·사건번호를
    # 옮기다 한 자리를 빠뜨리는 일이 잦았다(check_citations.py) → 0(항상 가장 확률 높은 토큰)과 비교한다
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--cases", default="data/cases.json", help="평가 전용 새 판례: data/heldout_cases.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    with open(args.cases, encoding="utf-8") as f:
        cases = [c for c in json.load(f) if strip_html(c.get("판례내용"))]
    texts = [strip_html(c["판례내용"]) for c in cases]

    llm = LLM(model=args.model, dtype="float16", tensor_parallel_size=args.tp, max_model_len=16384,
              gpu_memory_utilization=0.90, seed=0)
    # summarize.py(_ask)와 같은 생성 설정
    params = SamplingParams(temperature=args.temperature, max_tokens=1500, seed=0)

    start = time.perf_counter()
    summaries, outputs = run(llm, params, SEMANTIC_PROMPT, texts)
    # 12차 규칙: 1단계 결과가 거의 비었을 때만 "법리 + 적용" 프롬프트로 다시 시도
    retry = [i for i, s in enumerate(summaries) if len(s) < _EMPTY_RESULT_THRESHOLD]
    if retry:
        again, _ = run(llm, params, SEMANTIC_PROMPT_WITH_APPLICATION, [texts[i] for i in retry])
        for i, s in zip(retry, again):
            summaries[i] = s
    elapsed = time.perf_counter() - start

    results = []
    for c, s, o in zip(cases, summaries, outputs):
        results.append(c | {"ai_summary": s, "판결요지_clean": strip_html(c.get("판결요지")),
                            "_truncated": o.outputs[0].finish_reason == "length"})
    with open(f"data/summarized_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    stats = {"model": args.model, "tag": args.tag, "temperature": args.temperature, "cases": len(results), "fallback_used": len(retry),
             "truncated": sum(r["_truncated"] for r in results), "wall_time_s": round(elapsed, 1),
             "empty": sum(len(r["ai_summary"]) < _EMPTY_RESULT_THRESHOLD for r in results)}
    with open(f"data/run_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
