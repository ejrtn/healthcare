"""SFT 학습 데이터: 대화 → 4구역 JSON (기준선 JSON 모드와 같은 프롬프트).

train (79건)
  - ACI-BENCH train 67건
  - train_virtscribe_asr 12건: 같은 진료를 ASR 원본으로 전사한 버전. 정답 기록은 같고
    입력에 음성인식 잡음이 있어 ASR 강건성용 증강으로 쓴다.
    (train_aci_asrcorr 35건은 challenge train과 대화가 사실상 같아서 제외 — 29건 완전 동일,
     나머지도 유사도 0.9 이상. 평가 분할과 겹치는 기록은 0건.)
valid: ACI-BENCH valid 20건

출력: data/sft/{train,valid}.jsonl — {"id", "messages": [system, user, assistant]}
"""
import argparse
import csv
import json
from pathlib import Path

from prompts import build_messages
from schema import split_note

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ASR_TRAIN = ROOT / "data" / "raw" / "aci-bench" / "data" / "src_experiment_data" / "train_virtscribe_asr.csv"
OUT = ROOT / "data" / "sft"


def to_example(cid, dialogue, target):
    messages = build_messages(dialogue, json_mode=True)
    messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return {"id": cid, "messages": messages}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def token_stats(examples, tokenizer_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    total = sorted(len(tok.apply_chat_template(e["messages"], tokenize=True, return_dict=True)["input_ids"])
                   for e in examples)
    answer = sorted(len(tok(e["messages"][-1]["content"])["input_ids"]) for e in examples)
    n = len(total)
    return {"total_median": total[n // 2], "total_p95": total[int(n * 0.95)], "total_max": total[-1],
            "answer_median": answer[n // 2], "answer_max": answer[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct", help="토큰 길이 통계용")
    args = ap.parse_args()

    train = [to_example(r["id"], r["dialogue"], r["target"]) for r in load_jsonl(PROCESSED / "aci_train.jsonl")]
    with open(ASR_TRAIN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            train.append(to_example(f"{r['id']}_asr", r["dialogue"].strip(), split_note(r["note"].strip())))
    valid = [to_example(r["id"], r["dialogue"], r["target"]) for r in load_jsonl(PROCESSED / "aci_valid.jsonl")]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, examples in [("train", train), ("valid", valid)]:
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{name}: {len(examples)}건", token_stats(examples, args.tokenizer))


if __name__ == "__main__":
    main()
