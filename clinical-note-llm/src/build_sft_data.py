"""SFT 학습 데이터: 대화 → 4구역 JSON (기준선 JSON 모드와 같은 프롬프트).

train (79건)
  - ACI-BENCH train 67건
  - train_virtscribe_asr 12건: 같은 진료를 ASR 원본으로 전사한 버전. 정답 기록은 같고
    입력에 음성인식 잡음이 있어 ASR 강건성용 증강으로 쓴다.
    (train_aci_asrcorr 35건은 challenge train과 대화가 사실상 같아서 제외 — 29건 완전 동일,
     나머지도 유사도 0.9 이상. 평가 분할과 겹치는 기록은 0건.)
valid: ACI-BENCH valid 20건

출력: data/sft/{train,valid}.jsonl — {"id", "messages": [system, user, assistant]}

--grounded (→ data/sft_grounded/): 정답 기록에서 대화에 근거가 없는 나이·성(姓)을 지운다.
  정답 기록 작성자는 진료 정보 시스템을 보고 나이·성명을 채웠지만 모델은 대화만 본다. 그대로 학습하면
  모델이 "나이·성을 지어내는 것"을 배운다 — 1차 SFT 모델은 테스트 120건 중 54건에서 대화에 없는 나이를
  썼다(학습 전 0건, src/faithfulness.py).
"""
import argparse
import csv
import json
import re
from pathlib import Path

from faithfulness import spoken_numbers
from prompts import build_messages
from schema import split_note

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ASR_TRAIN = ROOT / "data" / "raw" / "aci-bench" / "data" / "src_experiment_data" / "train_virtscribe_asr.csv"
OUT = ROOT / "data" / "sft"


AGE = re.compile(r"\b(\d+)[- ]year[- ]old ")
# 정답 기록의 환자 소개 문장 "Steven Reed is a pleasant ..." 에서만 성명을 찾는다
# ("Physical Examination" 같은 대문자 두 단어를 이름으로 오인하지 않도록)
FULL_NAME = re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)(?= is an? )")
TITLED_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\. ([A-Z][a-z]+)\b")


def ground_text(text, dialogue):
    """대화에 없는 나이와 성을 지운다. 바뀐 항목 수도 돌려준다."""
    d_nums = spoken_numbers(dialogue)
    d_words = set(re.findall(r"[a-z]+", dialogue.lower()))
    changes = 0

    def age(m):
        nonlocal changes
        if int(m.group(1)) in d_nums:
            return m.group(0)
        changes += 1
        return ""

    def titled(m):  # "Ms. Scott": 대화에 성이 없으면 "The patient"
        nonlocal changes
        if m.group(1).lower() in d_words:
            return m.group(0)
        changes += 1
        return "the patient"

    text = AGE.sub(age, text)
    # "Steven Reed": 대화에 이름(steven)은 있고 성(reed)은 없으면, 기록 전체에서 그 성명을 이름만 남긴다
    for first, last in set(FULL_NAME.findall(text)):
        if first.lower() in d_words and last.lower() not in d_words:
            changes += text.count(f"{first} {last}")
            # "Mr. John Perry" → "John" (호칭 + 이름만은 어색하므로 호칭도 뗀다)
            text = re.sub(rf"\b(?:(?:Mr|Mrs|Ms|Miss)\. )?{first} {last}\b", first, text)
    text = TITLED_NAME.sub(titled, text)
    # 문장 첫머리로 온 "the patient"는 대문자로
    text = re.sub(r"(^|[.!?]\s+|\n)the patient", lambda m: m.group(1) + "The patient", text)
    return text, changes


def ground_target(target, dialogue):
    out, changes = {}, 0
    for k, v in target.items():
        out[k], c = ground_text(v, dialogue)
        changes += c
    return out, changes


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
    ap.add_argument("--grounded", action="store_true", help="대화에 없는 나이·성을 정답에서 지운다 → data/sft_grounded/")
    ap.add_argument("--no-stats", action="store_true", help="토큰 통계 생략 (토크나이저 다운로드 없이)")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "sft_grounded" if args.grounded else OUT
    changed = {"examples": 0, "edits": 0}

    def make(cid, dialogue, target):
        if args.grounded:
            target, n = ground_target(target, dialogue)
            changed["examples"] += n > 0
            changed["edits"] += n
        return to_example(cid, dialogue, target)

    train = [make(r["id"], r["dialogue"], r["target"]) for r in load_jsonl(PROCESSED / "aci_train.jsonl")]
    with open(ASR_TRAIN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            train.append(make(f"{r['id']}_asr", r["dialogue"].strip(), split_note(r["note"].strip())))
    valid = [make(r["id"], r["dialogue"], r["target"]) for r in load_jsonl(PROCESSED / "aci_valid.jsonl")]
    if args.grounded:
        print(f"근거 없는 나이·성 수정: {changed['examples']}건, {changed['edits']}곳")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in [("train", train), ("valid", valid)]:
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{name}: {len(examples)}건", "" if args.no_stats else token_stats(examples, args.tokenizer))


if __name__ == "__main__":
    main()
