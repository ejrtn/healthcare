"""되살린 차트(kaggle/08_extract_chart.py)를 걸러내고, "차트 + 대화" 학습·평가 데이터를 만든다. CPU에서 돈다.

추출 모델이 지시를 어길 수 있어 항목마다 규칙으로 한 번 더 거른다.
  - not_in_note: 숫자가 정답 기록에 없다 → 추출 모델이 지어낸 값
  - leak: 치료 계획·진단·진찰 소견처럼 이번 진료에서 정해진 내용 → 모델이 써야 할 답을 미리 보여 주는 셈

대화에 나온 정보도 차트에 그대로 둔다(실제 차트는 대화와 상관없이 같은 내용을 담는다).

정답 기록은 손대지 않는다(근거 정리 전 data/sft 그대로). 대화에 없던 나이·이름·수치가 이제 차트에 있으므로 지울 이유가 없다.

    python src/build_chart_data.py

출력: data/chart/{train,valid,test1,test2,test3}.jsonl — {"id", "chart", "chart_text", "dropped"}
      data/sft_chart/{train,valid}.jsonl — 학습용 messages, results/chart_stats.json
"""
import json
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


# 추출 모델이 빈칸 대신 써 넣은 말 ("N/A", "Past surgical history: None mentioned"), 값 없이 이름만 있는 활력징후
# ("Blood pressure"). "No known allergies"는 실제 정보라 남긴다
PLACEHOLDER = re.compile(
    r"^(?:[A-Za-z ]{1,30}:\s*)?(?:n/?a|none|unknown|not (?:mentioned|specified|stated|documented|provided)|"
    r"none (?:mentioned|reported|documented|specified)|no (?:specific )?allergies (?:mentioned|reported|documented)|"
    r"blood pressure|heart rate|pulse|respiratory rate|temperature|oxygen saturation|weight|height|bmi|vitals?)\.?$",
    re.I)


def numbers(text):
    return {int(n) for n in re.findall(r"\d+", text)}


def keep(item, note_nums):
    """차트 항목 하나를 남길지. (남김 여부, 뺀 이유)"""
    nums = numbers(item)
    if PLACEHOLDER.match(item.strip()):
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
            ok, why = keep(item, note_nums)
            (out[k].append(item) if ok else dropped.append((item, why)))
    # 이름은 기록에 그대로 있어야 하고, 나이는 숫자가 기록에 있어야 한다
    out["name"] = chart.get("name", "") if chart.get("name", "") and chart["name"] in note else ""
    age = chart.get("age", "")
    out["age"] = age if age and numbers(age) and numbers(age) <= note_nums else ""
    sex = chart.get("sex", "")
    out["sex"] = "" if PLACEHOLDER.match(sex.strip()) else sex
    return out, dropped


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
        reasons, kept, rows = Counter(), Counter(), []
        for cid, (dialogue, note, _) in cases.items():
            chart, dropped = clean_chart(raw.get(cid), note)
            reasons.update(why for _, why in dropped)
            kept.update(k for k in LIST_FIELDS for _ in chart[k])
            kept.update(k for k in ["name", "age"] if chart[k])
            rows.append({"id": cid, "chart": chart, "chart_text": render_chart(chart), "dropped": dropped})
        with open(ROOT / "data" / "chart" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {"cases": len(rows), "kept": dict(kept), "dropped": dict(reasons),
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
