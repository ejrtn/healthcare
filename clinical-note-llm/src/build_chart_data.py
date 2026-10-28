"""되살린 차트(kaggle/08_extract_chart.py)를 걸러내고, "차트 + 대화" 학습·평가 데이터를 만든다. CPU에서 돈다.

추출 모델이 지시를 어길 수 있어 항목마다 규칙으로 한 번 더 거른다.
  - not_in_note: 숫자가 정답 기록에 없다 → 추출 모델이 지어낸 값
  - leak: 치료 계획·진단·진찰 소견처럼 이번 진료에서 정해진 내용 → 모델이 써야 할 답을 미리 보여 주는 셈

대화에 나온 정보도 차트에 그대로 둔다(실제 차트는 대화와 상관없이 같은 내용을 담는다).

정답 기록은 손대지 않는다(근거 정리 전 data/sft 그대로). 대화에 없던 나이·이름·수치가 이제 차트에 있으므로 지울 이유가 없다.

나이·성별이 비면 채운다(fill_demographics). 실제 EHR에는 인적사항이 항상 있는데, 정답 기록에 나이가 없으면 복원할 수
없어 차트가 비었다 — 이 빈칸 때문에 차트 SFT 1차 모델이 나이 없는 29건 중 16건에서 나이를 지어냈다. 이름은 채우지 않는다:
대화에서 환자를 이름으로 부르는 경우가 많아("hi, sarah") 지어낸 이름이 대화와 어긋날 수 있다.
채우지 않은 원래 차트(chart_text_strict, 모르면 "not recorded")는 "모르면 쓰지 않는가"를 보는 평가용으로 남긴다.

    python src/build_chart_data.py

출력: data/chart/{train,valid,test1,test2,test3}.jsonl — {"id", "chart", "chart_text", "chart_text_strict", "synthetic", "dropped"}
      data/sft_chart/{train,valid}.jsonl — 학습용 messages, results/chart_stats.json
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from evaluate import load_jsonl, parse_prediction
from prompts import build_messages, render_chart

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "valid", "test1", "test2", "test3"]
LIST_FIELDS = ["vitals", "prior_results", "history", "medications", "allergies"]

LEAK = re.compile(
    r"\b(plan|planned|recommend\w*|will|advis\w*|refer\w*|follow[- ]?up|prescrib\w*|increas\w*|decreas\w*|"
    r"discontinu\w*|start(?:ed|ing)? (?:on|her|him)|diagnos\w*|assessment|impression|exam(?:ination)?|tender\w*|"
    r"palpat\w*|auscultat\w*|suspect\w*|order\w*|rule out|concern(?:ing)? for)\b", re.I)


# 추출 모델이 빈칸 대신 써 넣은 말 ("N/A", "Past surgical history: None mentioned", "Height: [not specified]",
# "Not specified in the note"), 값 없이 이름만 있는 활력징후("Blood pressure"). 이런 말이 섞이면 모르는 값을
# "not recorded"로 통일한 차트 형식(prompts.render_chart)이 깨진다. "No known allergies"는 실제 정보라 남긴다
PLACEHOLDER = re.compile(
    r"^(?:[A-Za-z ,()/&-]{1,40}:\s*)?\[?(?:n/?a|none|unknown|"
    r"not (?:mentioned|specified|stated|documented|provided|available|recorded)(?: (?:in|from) the (?:note|conversation|chart))?|"
    r"none (?:mentioned|reported|documented|specified)|no (?:specific )?allergies (?:mentioned|reported|documented)|"
    r"blood pressure|heart rate|pulse|respiratory rate|temperature|oxygen saturation|weight|height|bmi|vitals?)\]?\.?$",
    re.I)


# 위 목록으로 다 못 잡는 변형("[REDACTED - not provided in the conversation or note]", "Allergies not specified in the
# note.")은 차트 내용이 아니라 추출 작업에 대한 말이다 — 실제 차트는 "노트"나 "대화"를 언급하지 않는다
META = re.compile(r"\b(?:in|from) the (?:note|conversation)\b|redacted|\[[^\]]*not provided[^\]]*\]|"
                  r"^[^:]*\bnot (?:specified|provided|mentioned|documented)\b[^:]*\.?$", re.I)
# "Right ankle x-ray: No fracture (date not specified)" → 괄호 안 군말만 지운다
META_PAREN = re.compile(r"\s*\([^)]*\bnot (?:specified|provided|included|mentioned)\b[^)]*\)", re.I)
SEX = {"female": "Female", "f": "Female", "woman": "Female", "male": "Male", "m": "Male", "man": "Male"}


def numbers(text):
    return {int(n) for n in re.findall(r"\d+", text)}


def keep(item, note_nums):
    """차트 항목 하나를 남길지. (남김 여부, 뺀 이유)"""
    nums = numbers(item)
    if PLACEHOLDER.match(item.strip()) or META.search(item):
        return False, "placeholder"
    if LEAK.search(item):
        return False, "leak"
    if nums and not nums <= note_nums:
        return False, "not_in_note"
    return True, ""


def clean_chart(chart, note):
    if chart is None:
        return {"name": "", "age": "", "sex": ""} | {k: [] for k in LIST_FIELDS}, [("all", "extract_failed")]
    note_nums = numbers(note)
    out, dropped = {}, []
    for k in LIST_FIELDS:
        out[k] = []
        for item in chart.get(k, []):
            item = META_PAREN.sub("", item).strip()
            ok, why = keep(item, note_nums)
            (out[k].append(item) if ok else dropped.append((item, why)))
    # 이름은 기록에 그대로 있어야 하고, 나이는 숫자가 기록에 있어야 한다
    out["name"] = chart.get("name", "") if chart.get("name", "") and chart["name"] in note else ""
    age = chart.get("age", "")
    out["age"] = age if age and numbers(age) and numbers(age) <= note_nums else ""
    # "F", "female", "Female"이 섞이지 않게 통일. 그 밖의 값(Non-binary 등)은 기록에 있을 때만 그대로 둔다
    sex = chart.get("sex", "").strip()
    out["sex"] = SEX.get(sex.lower(), sex if sex and sex.lower() in note.lower() else "")
    return out, dropped


# 기록 맥락과 어긋나지 않는 나이 범위 (학생에게 70세를, 심근경색 병력에 20대를 붙이지 않도록). 앞의 규칙이 우선한다.
# 대화는 보지 않는다 — "고등학교 때 담배를 시작했다", "만성 신장병은 아닌지 보자" 같은 말에 잘못 걸렸다.
AGE_RANGES = [
    (re.compile(r"\b(?:is|currently|she's) pregnant\b"), (24, 38)),
    (re.compile(r"\b(?:student|college|university)\b"), (18, 24)),
    (re.compile(r"\b(?:retired|grandchild\w*|grandkid\w*|grandson|granddaughter)\b"), (65, 82)),
    (re.compile(r"\b(?:coronary|myocardial|stemi|stent|heart failure|chf|copd|stage (?:3|iii|4|iv) chronic|"
                r"arthroplasty|osteoarthritis|macular degeneration)\b"), (52, 80)),
    (re.compile(r"\b(?:diabet\w*|hypertension|high blood pressure|gout|hyperlipidemia)\b"), (40, 75)),
]


def fill_demographics(chart, cid, dialogue, note, llm=None):
    """비어 있는 이름·나이·성별을 채운다. (채운 차트, 채운 항목 목록).

    llm: kaggle/09_fill_demographics.py가 대화·기록 내용을 읽고 채운 값. 있으면 그것을 쓰고, 없으면(그 단계를 아직
    안 돌렸으면) 아래 규칙으로 나이·성별만 채운다(이름은 대화와 어긋날 수 있어 규칙으로는 채우지 않는다).
    규칙은 사례 id로 시드를 정해 다시 돌려도 같은 값이 나온다.
    """
    chart, filled = dict(chart), []
    if llm and llm.get("name"):
        for key in ["name", "age", "sex"]:
            if not chart[key] and llm.get(key):
                chart[key] = str(llm[key])
                filled.append(key)
        return chart, filled
    if not chart["age"]:
        context = note.lower()
        low, high = next((r for pat, r in AGE_RANGES if pat.search(context)), (28, 75))
        chart["age"] = str(random.Random(cid).randint(low, high))
        filled.append("age")
    if not chart["sex"]:
        she = len(re.findall(r"\b(?:she|her|hers)\b", note, re.I))
        he = len(re.findall(r"\b(?:he|him|his)\b", note, re.I))
        if she != he:  # 기록의 대명사로 드러나는 경우만
            chart["sex"] = "Female" if she > he else "Male"
            filled.append("sex")
    return chart, filled


def sources():
    """분할별 {id: (대화, 정답 기록, 정답 JSON 문자열 또는 None)}."""
    src = {}
    for split in ["train", "valid"]:
        src[split] = {r["id"]: (r["messages"][1]["content"], parse_prediction(r["messages"][-1]["content"])[0],
                                r["messages"][-1]["content"])
                      for r in load_jsonl(ROOT / "data" / "sft" / f"{split}.jsonl")}
    for split in ["test1", "test2", "test3"]:
        src[split] = {c["id"]: (c["dialogue"], c["note"], None)
                      for c in load_jsonl(ROOT / "data" / "processed" / f"aci_{split}.jsonl")}
    return src


def main():
    stats = {}
    (ROOT / "data" / "sft_chart").mkdir(parents=True, exist_ok=True)
    for split, cases in sources().items():
        raw = {r["id"]: r["chart"] for r in load_jsonl(ROOT / "data" / "chart" / f"raw_{split}.jsonl")}
        demo_path = ROOT / "data" / "chart" / f"demographics_{split}.jsonl"
        demo = {r["id"]: r for r in load_jsonl(demo_path)} if demo_path.exists() else {}
        reasons, kept, synthetic, rows = Counter(), Counter(), Counter(), []
        for cid, (dialogue, note, _) in cases.items():
            strict, dropped = clean_chart(raw.get(cid), note)
            chart, filled = fill_demographics(strict, cid, dialogue, note, demo.get(cid))
            reasons.update(why for _, why in dropped)
            kept.update(k for k in LIST_FIELDS for _ in chart[k])
            kept.update(k for k in ["name", "age"] if strict[k])
            synthetic.update(filled)
            rows.append({"id": cid, "chart": chart, "chart_text": render_chart(chart), "chart_strict": strict,
                         "chart_text_strict": render_chart(strict), "synthetic": filled,
                         "synthetic_source": (demo.get(cid) or {}).get("source"), "dropped": dropped})
        with open(ROOT / "data" / "chart" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {"cases": len(rows), "kept": dict(kept), "dropped": dict(reasons), "synthetic": dict(synthetic),
                        "empty_charts": sum(not any(r["chart"].values()) for r in rows)}

        if split in ("train", "valid"):
            with open(ROOT / "data" / "sft_chart" / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    dialogue, _, target_json = cases[r["id"]]
                    messages = build_messages(dialogue, json_mode=True, chart_text=r["chart_text"])
                    messages.append({"role": "assistant", "content": target_json})
                    f.write(json.dumps({"id": r["id"], "messages": messages}, ensure_ascii=False) + "\n")

    (ROOT / "results" / "chart_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
