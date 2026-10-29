"""근거 검사: 생성된 진료기록에 대화에 없는 내용이 얼마나 들어갔는지 자동으로 센다.

ROUGE는 정답과 단어가 얼마나 겹치는지만 봐서, 대화에 없는 사실을 지어내도 정답 문체와 비슷하면 점수가
오른다. 사례를 직접 읽어 보니 SFT 모델이 나이·성(姓)·검사 결과를 지어내는 경우가 많아, 확인하기 쉬운
두 가지를 120건 전체에서 센다.

  1. 근거 없는 숫자: 기록에 나온 숫자(용량, 검사 수치, 기간 등. 나이는 3번에서 따로)가 대화에 숫자로도, 영어 단어로도
     (예: 대화 "forty milligrams" ↔ 기록 "40 mg") 나오지 않는 경우
  2. 근거 없는 검사: 기록에 MRI·CT·심초음파 등 검사 이름이 나오는데 대화에는 그 검사가 한 번도 언급되지 않은 경우
  3. 지어낸 나이: "NN-year-old"의 나이가 대화에 없는 경우 (정답 기록에는 대화에 없는 나이가 자주 들어 있다 —
     기록 작성자가 진료 정보 시스템을 보고 채운 것으로 보인다)

정답 기록 자체도 같은 기준으로 재서, "정답에도 대화에 없는 내용이 이만큼 있다"는 기준선으로 쓴다.

    python src/faithfulness.py outputs/baseline/Qwen2.5-3B-Instruct-fp0.4_json.jsonl outputs/sft/qwen3b-r16-fp0.4_json.jsonl
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_SPLITS = ["test1", "test2", "test3"]

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
# 서수는 기수로 읽는다: "the fifth mtp joint" ↔ 기록 "5th", "october thirty first" ↔ "31st"
ORDINALS = {"first": "one", "second": "two", "third": "three", "fourth": "four", "fifth": "five", "sixth": "six",
            "seventh": "seven", "eighth": "eight", "ninth": "nine", "tenth": "ten", "eleventh": "eleven",
            "twelfth": "twelve", "twentieth": "twenty", "thirtieth": "thirty"}
ORDINALS |= {u + "th": u for u in UNITS[13:20]}  # thirteenth ~ nineteenth

# 검사 이름 → 대화에서 그 검사를 가리킬 수 있는 표현들
TESTS = {
    "mri": ["mri"],
    "ct": [r"\bct\b", "cat scan"],
    "x-ray": ["x-ray", "xray", "x ray", "radiograph"],
    "echocardiogram": ["echo"],
    "ekg": ["ekg", "ecg", "electrocardiogram"],
    "ultrasound": ["ultrasound", "sonogram"],
    "colonoscopy": ["colonoscopy"],
    "biopsy": ["biopsy"],
}


def word_value(words, i):
    """words[i]부터 읽히는 1~99 값과 쓴 단어 수: "sixty five" → (65, 2), "fifteen" → (15, 1). 없으면 (None, 0)."""
    if i >= len(words):
        return None, 0
    w = words[i]
    if w in TENS:
        if i + 1 < len(words) and words[i + 1] in UNITS[1:10]:
            return TENS[w] + UNITS.index(words[i + 1]), 2
        return TENS[w], 1
    if w in UNITS[1:]:
        return UNITS.index(w), 1
    return None, 0


def spoken_numbers(text):
    """대화 속 숫자 표현(숫자 + 영어 단어)을 정수 집합으로. "one twenty over seventy" 같은 표현을 위해 단순 조합도 넣는다.

    근거 검사 v2 (2026-10-29): 표시된 51문장을 대조해 보니 21개가 오탐이었고, 대부분 아래 표현을 못 읽어서였다.
    "1,792" · "a hundred and fifteen" · "three hundred and sixty" · "one seventy five" · "nine one one" · "two and a half"
    서수와 연도(2026-10-29, API 끝까지 시험에서): "the fifth mtp joint" · "october thirty first" · "twenty twenty one"
    """
    nums = {int(n) for n in re.findall(r"\d+", text)}
    nums |= {int(n.replace(",", "")) for n in re.findall(r"\d{1,3}(?:,\d{3})+", text)}  # "1,792"
    words = [ORDINALS.get(w, w) for w in re.findall(r"[a-z]+|\d+", text.lower())]
    nums |= {i for i, w in enumerate(UNITS) if w in words}
    for i, w in enumerate(words):
        v, _ = word_value(words, i)
        if v is not None:
            nums.add(v)
        for scale, size in (("hundred", 100), ("thousand", 1000)):  # "one thousand milligrams" = 1000
            if w == scale:
                prev = words[i - 1] if i > 0 else ""
                base = UNITS.index(prev) if prev in UNITS else int(prev) if prev.isdigit() else 1
                nums.add(base * size)
                j = i + 2 if i + 1 < len(words) and words[i + 1] == "and" else i + 1
                rest, _ = word_value(words, j)  # "a hundred and fifteen" = 115, "three hundred sixty" = 360
                if rest is not None:
                    nums.add(base * size + rest)
        if (w in TENS or w in UNITS[10:]) and i + 1 < len(words) and (words[i + 1] in TENS or words[i + 1] in UNITS[10:]):
            # 연도: "twenty twenty one" = 2021, "nineteen ninety" = 1990
            head = TENS[w] if w in TENS else UNITS.index(w)
            if (words[i - 1] if i > 0 else "") not in TENS:  # "sixty five" 다음의 "twenty"부터 다시 읽지 않는다
                nums.add(head * 100 + word_value(words, i + 1)[0])
        if w in UNITS[1:10] and i + 1 < len(words) and (words[i + 1] in TENS or words[i + 1] in UNITS[10:]):
            # "one twenty" = 120, "one fifteen" = 115, "one seventy five" = 175
            nxt, _ = word_value(words, i + 1)
            nums.add(UNITS.index(w) * 100 + nxt)
        if w in UNITS[:10]:  # 숫자를 한 자리씩 읽은 것: "nine one one" = 911 (3~4자리)
            digits = []
            for x in words[i:i + 4]:
                if x not in UNITS[:10]:
                    break
                digits.append(str(UNITS.index(x)))
                if len(digits) >= 3:
                    nums.add(int("".join(digits)))
        if w == "and" and words[i + 1:i + 3] == ["a", "half"] and i > 0:  # "two and a half" → 2.5 (소수 조건 {25})
            whole = UNITS.index(words[i - 1]) if words[i - 1] in UNITS else words[i - 1]
            if str(whole).isdigit():
                nums.add(int(f"{whole}5"))
    return nums


def note_numbers(text):
    """기록에 나온 숫자와, 그 숫자가 대화에 근거가 있다고 볼 조건들(하나라도 전부 충족되면 근거 있음).

    소수 6.8은 대화에 "six point eight"(6과 8 둘 다) 또는 "68"이 있어야 한다. 섹션 번호 같은 1~3은 흔해서 뺀다.
    """
    out = []
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        tok = m.group()
        if "." in tok:
            whole, frac = tok.split(".")
            out.append((tok, [{int(whole), int(frac)}, {int(whole + frac)}]))
        elif int(tok) > 3:
            out.append((tok, [{int(tok)}]))
    return out


def check(note, dialogue):
    d = dialogue.lower()
    d_nums = spoken_numbers(dialogue)
    # 나이는 따로 센다(3번). 병명 속 숫자(COVID-19, type 2)는 사실 주장이 아니라 뺀다
    rest = re.sub(r"\d+[- ]year[- ]old|covid[- ]?19|type (?:1|2|one|two)\b", " ", note, flags=re.I)
    rest = re.sub(r"(?m)^[ \t]*[-•]?[ \t]*\d+\.(?=\s|$)", " ", rest)  # 목록 번호 "4."
    # 척도 표기: 통증 "8/10"·"3 out of 10", 심잡음 "grade 2/6"의 분모는 표기 관습이다. 분자는 1~3이어도 확인한다
    scales = []

    def _scale(m):
        scales.append((m.group(1), [{int(m.group(1))}]))
        return " "
    rest = re.sub(r"\b(\d+)\s*(?:/|out of)\s*10\b", _scale, rest)
    rest = re.sub(r"(?<=grade )(\d)\s*/\s*6\b", _scale, rest, flags=re.I)
    # 연대 표현 "in the lower 90s": 그 범위의 숫자가 대화에 있으면 근거 있음 (대화 "ninety four" ↔ 기록 "90s")
    decades = [(m.group(), [{n}]) for m in re.finditer(r"\b[1-9]0(?=s\b)", rest)
               for n in [next((x for x in sorted(d_nums) if int(m.group()) <= x < int(m.group()) + 10), int(m.group()))]]
    rest = re.sub(r"\b[1-9]0s\b", " ", rest)
    numbers = scales + decades + note_numbers(rest)
    unsupported_nums = [tok for tok, conds in numbers if not any(c <= d_nums for c in conds)]
    n = note.lower()
    unsupported_tests = [name for name, pats in TESTS.items()
                         if re.search(re.escape(name) if name != "ct" else r"\bct\b", n)
                         and not any(re.search(p, d) for p in pats)]
    ages = re.findall(r"(\d+)[- ]year[- ]old", note)
    invented_age = [a for a in ages if int(a) not in d_nums]
    return {"unsupported_numbers": unsupported_nums, "unsupported_tests": unsupported_tests,
            "invented_age": invented_age, "numbers_total": len(numbers)}


def summarize(results):
    n = len(results)
    total_nums = sum(r["numbers_total"] for r in results)
    bad_nums = sum(len(r["unsupported_numbers"]) for r in results)
    return {
        "cases": n,
        "numbers_total": total_nums,
        "unsupported_numbers": bad_nums,
        "unsupported_number_rate": round(100 * bad_nums / max(1, total_nums), 1),
        "cases_with_unsupported_number": sum(bool(r["unsupported_numbers"]) for r in results),
        "cases_with_unsupported_test": sum(bool(r["unsupported_tests"]) for r in results),
        "cases_with_invented_age": sum(bool(r["invented_age"]) for r in results),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="예측 jsonl (outputs/...)")
    ap.add_argument("--out", default=str(ROOT / "results" / "faithfulness.json"))
    ap.add_argument("--chart-dir", help="data/chart: 차트도 근거로 본다 ('차트 + 대화'로 생성한 예측용)")
    ap.add_argument("--chart-field", default="chart_text", help="생성 때 준 차트와 같은 것 (01_baseline.py --chart-field)")
    args = ap.parse_args()
    # 채점 모듈(rouge_score 의존)은 CLI에서만 쓴다 — API 서버는 check()만 가져다 쓴다
    from evaluate import load_jsonl, parse_prediction

    cases = [g for s in TEST_SPLITS for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]
    if args.chart_dir:  # 근거 = 차트 + 대화. 정답 기록도 같은 근거로 재서 기준선이 함께 바뀐다
        charts = {r["id"]: r[args.chart_field] for s in TEST_SPLITS for r in load_jsonl(ROOT / args.chart_dir / f"{s}.jsonl")}
        cases = [c | {"dialogue": charts[c["id"]] + "\n\n" + c["dialogue"]} for c in cases]
    report = {"gold": None, "systems": {}}
    gold = [check(c["note"], c["dialogue"]) | {"id": c["id"]} for c in cases]
    report["gold"] = summarize(gold)
    rows = [("정답 기록 (기준선)", report["gold"])]
    for path in args.preds:
        preds = {p["id"]: p["prediction"] for p in load_jsonl(ROOT / path)}
        per_case = []
        for c in cases:
            note, _ = parse_prediction(preds.get(c["id"], ""))
            per_case.append(check(note, c["dialogue"]) | {"id": c["id"]})
        s = summarize(per_case)
        report["systems"][path] = {"summary": s, "per_case": per_case}
        rows.append((Path(path).stem, s))

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("| 시스템 | 숫자 중 근거 없음 | 근거 없는 숫자가 있는 기록 | 근거 없는 검사가 있는 기록 | 지어낸 나이가 있는 기록 |")
    print("|---|---|---|---|---|")
    for name, s in rows:
        print(f"| {name} | {s['unsupported_numbers']}/{s['numbers_total']} ({s['unsupported_number_rate']}%) | "
              f"{s['cases_with_unsupported_number']}/{s['cases']} | {s['cases_with_unsupported_test']}/{s['cases']} | "
              f"{s['cases_with_invented_age']}/{s['cases']} |")


if __name__ == "__main__":
    main()
