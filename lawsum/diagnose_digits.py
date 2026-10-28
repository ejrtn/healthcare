"""A.X가 숫자 한 자리를 빠뜨리는 원인 가르기 (Kaggle).

temperature 0에서도 같은 자리가 똑같이 빠져 샘플링 탓은 아니다. 입력 토큰화도 정상이다(로컬 확인). 남은 후보:
  1. vLLM이 출력 토큰을 글자로 바꾸는(detokenize) 과정의 문제 → 출력 토큰 번호를 HF 토크나이저로 다시 풀어 비교
  2. FP16 정밀도 문제(A.X는 BF16으로 학습, T4는 BF16 미지원) → 같은 입력을 HF transformers로
     GPU FP16과 CPU BF16(Kaggle RAM 30GB)에서 각각 생성해 비교
틀린 표기가 앞쪽(출력 210~430토큰)에 나오는 3건으로, 거기까지만 생성한다.

    !python diagnose_digits.py --mode vllm
    !python diagnose_digits.py --mode hf-gpu-fp16
    !python diagnose_digits.py --mode hf-cpu-bf16     # 느리다(건당 수 분)

결과: data/diagnose_<mode>.json
"""
import argparse
import json

from check_citations import check
from summarize import SEMANTIC_PROMPT, strip_html

MODEL = "skt/A.X-4.0-Light"
# 사건번호: 틀린 표기가 나오는 위치(출력 토큰)보다 넉넉히 생성
CASES = {"618229": 450, "616245": 300, "618185": 520}


def load():
    cases = {str(c["_판례일련번호"]): c for c in json.load(open("data/cases.json", encoding="utf-8"))}
    return {cid: strip_html(cases[cid]["판례내용"]) for cid in CASES}


def messages(src):
    return [{"role": "system", "content": SEMANTIC_PROMPT}, {"role": "user", "content": src}]


def report(mode, rows):
    for r in rows:
        r["missing"] = {k: v["missing"] for k, v in check(r["text"], r["source"]).items() if v["missing"]}
        if "hf_decoded" in r:
            r["missing_hf_decoded"] = {k: v["missing"] for k, v in check(r["hf_decoded"], r["source"]).items()
                                       if v["missing"]}
            r["decode_mismatch"] = r["hf_decoded"] != r["text"]
        del r["source"]
        print(r["id"], json.dumps({k: v for k, v in r.items() if k not in ("text", "hf_decoded")}, ensure_ascii=False))
    json.dump(rows, open(f"data/diagnose_{mode}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def run_vllm(sources):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="float16", tensor_parallel_size=2, max_model_len=16384, seed=0)
    rows = []
    for cid, src in sources.items():
        o = llm.chat([messages(src)], SamplingParams(temperature=0, max_tokens=CASES[cid]))[0].outputs[0]
        rows.append({"id": cid, "source": src, "text": o.text,
                     "hf_decoded": tok.decode(list(o.token_ids), skip_special_tokens=True)})
    return rows


def run_hf(sources, device, dtype_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype,
                                                 device_map="auto" if device == "gpu" else "cpu")
    rows = []
    for cid, src in sources.items():
        inputs = tok.apply_chat_template(messages(src), add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=CASES[cid], do_sample=False)
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({"id": cid, "source": src, "text": text})
        print(cid, "done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vllm", "hf-gpu-fp16", "hf-cpu-bf16"], required=True)
    args = ap.parse_args()
    sources = load()
    if args.mode == "vllm":
        rows = run_vllm(sources)
    elif args.mode == "hf-gpu-fp16":
        rows = run_hf(sources, "gpu", "fp16")
    else:
        rows = run_hf(sources, "cpu", "bf16")
    report(args.mode, rows)


if __name__ == "__main__":
    main()
