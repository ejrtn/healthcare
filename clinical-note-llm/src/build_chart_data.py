"""되살린 차트(kaggle/08_extract_chart.py → data/chart/extracted/, 고치지 않음)를 걸러내고, "차트 + 대화" 학습·평가 데이터를 만든다. CPU에서 돈다.

추출 모델이 지시를 어길 수 있어 항목마다 규칙으로 한 번 더 거른다.
  - not_in_note: 숫자가 정답 기록에 없다 → 추출 모델이 지어낸 값
  - leak: 치료 계획·진단·진찰 소견처럼 이번 진료에서 정해진 내용 → 모델이 써야 할 답을 미리 보여 주는 셈
  - 이번 방문의 검사(same_day): 진료 직전에 찍은 엑스레이처럼 영상은 진료 전에 있지만, 판독 결과("no fracture")는 의사가
    진료 중에 내린 판단이고 정답 RESULTS 문장 그대로다. 결과를 지우고 "X-rays of the right knee: obtained today, not yet
    reviewed"처럼 촬영 사실만 남긴다. 검사 항목과 그 항목을 옮겨 온 정답 기록 문장을 함께 본다. 엑스레이는 날짜·외부 병원·
    응급실·작년 같은 표시가 없으면 이번 방문 것으로 본다(ACI-BENCH 대화에서 "we did an x-ray before you came in"이 대부분)
  - new_rx: 이번 진료에서 처방한 약("i'm gon na prescribe you some oxycodone") → 복용 중인 약이 아니다
  - exam: 활력징후 칸에 들어간 진찰 소견("Neck: supple", "Capillary refill is brisk")
  - assessment: 과거력 칸에 들어간 이번 진료의 판단("Hypertension: Uncontrolled at this time")

대화에 나온 정보도 차트에 그대로 둔다(실제 차트는 대화와 상관없이 같은 내용을 담는다).

정답 기록은 손대지 않는다(근거 정리 전 data/sft 그대로). 대화에 없던 나이·이름·수치가 이제 차트에 있으므로 지울 이유가 없다.

빈칸 채우기:
  - 이름·나이·성별: 실제 EHR에는 항상 있으므로 항상 채운다. 빈칸 때문에 차트 SFT 1차 모델이 나이 없는 29건 중 16건에서
    나이를 지어냈다. LLM 판단 없이 규칙(src/demographics.py)으로만 채우고 값마다 출처를 남긴다(demographics_source).
    추출 모델이 뽑은 값과 달라도 규칙 값을 쓴다 — 같은 데이터면 늘 같은 값이 나와야 하기 때문이다.
  - 그 밖의 항목: 대화가 가리키는데 차트에 없던 것만 더한다(add_mentioned, kaggle/09_complete_chart.py의 additions가
    있을 때). 정답 기록에 값이 있으면 그 값, 없으면 지어낸 값.
채우지 않은 원래 차트(chart_text_strict, 모르면 빈칸)는 "모르면 쓰지 않는가"를 보는 평가용으로 남긴다.

    python src/build_chart_data.py [--sft-dir data/sft_chart_v2]

출력: data/chart/{train,valid,test1,test2,test3}.jsonl — {"id", "chart", "chart_text", "chart_text_strict", "synthetic", "added", "dropped"}
      data/sft_chart/{train,valid}.jsonl — 학습용 messages, results/chart_stats.json
"""
import argparse
import json
import re
import zlib
from collections import Counter
from pathlib import Path

from demographics import DIALOGUE_INTRO, dialogue_ages, fill
from evaluate import load_jsonl, parse_prediction
from prompts import build_messages, render_chart

ROOT = Path(__file__).resolve().parent.parent
# LLM 출력(kaggle/08·09)은 여기에 그대로 두고 읽기만 한다. 이 스크립트의 결과는 data/chart/, data/sft_chart/에 새로 쓴다
EXTRACTED = ROOT / "data" / "chart" / "extracted"
SPLITS = ["train", "valid", "test1", "test2", "test3"]
LIST_FIELDS = ["vitals", "prior_results", "history", "medications", "allergies"]

LEAK = re.compile(
    r"\b(plan|planned|recommend\w*|will|advis\w*|refer\w*|follow[- ]?up|prescrib\w*|increas\w*|decreas\w*|"
    r"discontinu\w*|start(?:ing)?|new(?:ly)? prescri\w*|diagnos\w*|assessment|impression|exam(?:ination)?|tender\w*|"
    r"palpat\w*|auscultat\w*|suspect\w*|order\w*|rule out|concern(?:ing)? for)\b", re.I)


# 검사를 오늘 했다는 표현만 잡는다 — "presents to the clinic today"처럼 방문만 말하는 문장은 걸리지 않게 검사 동사에 붙은 것만
_TODAY = r"(?:today|in (?:the )?office|in clinic|during (?:this|the) visit|on today's visit|at this visit)"
SAME_DAY = re.compile(
    rf"\b(?:obtain\w*|taken|performed|done|completed|order\w*)\b[^.;]{{0,50}}\b{_TODAY}|"
    rf"\b(?:x-?rays?|imaging|ekg|ecg|test)\s+today\b|\btoday,? (?:\d+ views|reveals?|shows?|demonstrates?)\b|"
    rf"\b(?:x-?rays?|ultrasound|test|scan)\b[^.;]{{0,40}}\b(?:during (?:this|the) visit|at this visit)", re.I)


def source_sentence(item, note):
    """항목을 옮겨 온 정답 기록 문장 (단어가 두 개 이상 겹치는 문장 중 가장 많이 겹치는 것). 없으면 ""."""
    words = set(re.findall(r"[a-z]{3,}|\d+", item.lower()))  # 3글자: "EKG", "CBC"
    best, best_n = "", 1
    for s in re.split(r"(?<=[.!?])\s+|\n+", note):
        n = len(words & set(re.findall(r"[a-z]{3,}|\d+", s.lower())))
        if n > best_n:
            best, best_n = s, n
    return best


IMAGING = re.compile(r"\bx-?rays?\b|\bxrays?\b|\bradiographs?\b|\b\d+ views\b", re.I)
PRIOR = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}|\boutside\b|\bemergency\b|\b(?:er|ed)\b|\blast (?:year|month|week)\b|"
                   r"\bprevious(?:ly)?\b|\bprior\b|\bago\b|\bin (?:19|20)\d\d\b", re.I)


def same_day(item, note):
    # "obtained at an outside facility, is reviewed today": 다른 병원에서 미리 찍은 검사는 진료 전 차트에 있다
    src = source_sentence(item, note)
    if PRIOR.search(item) or re.search(r"\boutside\b", src, re.I):
        return bool(SAME_DAY.search(item))
    return bool(SAME_DAY.search(item) or SAME_DAY.search(src) or (IMAGING.search(item) and not PRIOR.search(src)))


RESULT_START = re.compile(r"\s*(?::|,|\b(?:reveals?|revealed|shows?|showed|demonstrat\w*|is|are|was|were|does|do|obtained|taken|"
                          r"performed|done|completed|reviewed|ordered|today)\b)", re.I)


def test_only(item):
    """판독 결과를 지우고 검사 이름만 남긴다. "X-rays of the right knee taken in office today reveal..." →
    "X-rays of the right knee: obtained today, not yet reviewed"."""
    m = RESULT_START.search(item)
    name = (item[:m.start()] if m and m.start() > 0 else item).strip(" .,;")
    if re.match(r"(?:no|normal|negative)\b", name, re.I):  # "No fractures found in x-rays taken during this visit"
        name = IMAGING.search(item).group() if IMAGING.search(item) else "Test"
        name = name[0].upper() + name[1:]
    return f"{name}: obtained today, not yet reviewed"


def planned_test(item, note):
    """오늘 하기로 한 검사("We will obtain a kidney ultrasound today", "Order for hemoglobin A1c ... provided today")는
    아직 결과가 없는 이번 진료의 계획이다."""
    src = source_sentence(item, note)
    return bool(re.search(r"\b(?:will|order\w*)\b", src, re.I)
                and not re.search(r"\b(?:obtained|taken|performed|done|completed)\b", src, re.I))


# 의사가 지금 처방하겠다고 말한 바로 뒤에 약 이름이 나오면 이번 진료의 새 처방이다("i'm gon na prescribe you some oxycodone",
# "we should go ahead and put you on protonix"). "dr. brown put you on buspar", "she was prescribed"처럼 과거 처방은 아니다
PRESCRIBE = re.compile(r"\b(?:i'm (?:also )?gon na|i am gon na|i'm going to|we're going to|i will|we will|we'll|"
                       r"i (?:also )?wan na|i want to|let's|we should|go ahead and|and we)\b[^.?]{0,40}?"
                       r"(?:\b(?:prescrib\w*|put you on|start you on|get you on|give you)\b[^.?]{0,60}|\btry (?:some |the )?)$")
GENERIC = {"medication", "medications", "blood", "antibiotics", "current", "pain", "other"}


def new_prescription(item, dialogue):
    drug = re.match(r"[A-Za-z][A-Za-z-]{2,}", item)
    if not drug or drug.group().lower() in GENERIC:
        return False
    d = dialogue.lower()
    for m in re.finditer(re.escape(drug.group().lower()), d):
        before = d[max(0, m.start() - 120):m.start()]
        if PRESCRIBE.search(before) and not re.search(r"instead of (?:the )?$", before):
            return True
    return False


# 활력징후 칸에 진찰 소견이 들어간 경우만 뺀다(검사 수치가 잘못 들어간 "Hemoglobin: 8.2"는 진료 전 정보라 남긴다)
EXAM = re.compile(r"\b(?:neck|heart(?! rate)|lungs?|musculoskeletal|capillary refill|murmur|edema|supple|wheez\w*|rales|"
                  r"abdomen|extremit\w*|skin|neuro\w*|tender\w*|range of motion)\b", re.I)
ASSESSMENT = re.compile(r"\bat this time\b|\buncontrolled\b|\bdoing well\b|\bhigh risk medication use\b", re.I)
FAMILY = re.compile(r"\b(?:mother|father|sister|brother|son|daughter|family)\b", re.I)  # 가족력의 "doing well"은 진단이 아니다


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
META = re.compile(r"\b(?:in|from) (?:the )?(?:note|conversation)\b|redacted|\[[^\]]*not provided[^\]]*\]|"
                  r"^[^:]*\bnot (?:specified|provided|mentioned|documented)\b[^:]*\.?$", re.I)
# "Right ankle x-ray: No fracture (date not specified)" → 괄호 안 군말만 지운다
META_PAREN = re.compile(r"\s*\([^)]*\bnot (?:specified|provided|included|mentioned)\b[^)]*\)", re.I)
SEX = {"female": "Female", "f": "Female", "woman": "Female", "male": "Male", "m": "Male", "man": "Male"}


# 대괄호 자리 표시("[value]", "[details]", "[date]", "[current dosage]", "[list]")는 추출 모델이 빈칸에 써 넣은 것이다.
# 실제 차트에는 없는 표기라 문구를 하나씩 목록에 넣는 대신 대괄호는 모두 지운다 — 학습 전 모델이 차트의 "[value]"를
# 진료기록에 그대로 옮겨 적었다(Qwen2.5-3B + 차트, 2건).
BRACKET = re.compile(r"\s*\[[^\]]*\]")
EMPTY_PAREN = re.compile(r"\s*\(\s*(?:[A-Za-z ]{1,20}:\s*)?\)")
LABEL_ONLY = re.compile(r"^[A-Za-z0-9 ,()/&.-]{1,40}:\s*$")


def strip_placeholders(item):
    """항목에서 대괄호 자리 표시를 지우고, 그 때문에 값 없이 이름만 남은 조각("Past Surgical History:")도 지운다."""
    parts = []
    for part in item.split(";"):
        part = EMPTY_PAREN.sub("", BRACKET.sub("", part)).strip()
        if part and not LABEL_ONLY.match(part):
            parts.append(part)
    return "; ".join(parts)


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


def clean_chart(chart, note, dialogue=""):
    if chart is None:
        return {"name": "", "age": "", "sex": ""} | {k: [] for k in LIST_FIELDS}, [("all", "extract_failed")]
    note_nums = numbers(note)
    out, dropped = {}, []
    for k in LIST_FIELDS:
        out[k] = []
        for raw_item in chart.get(k, []):
            if META.search(raw_item):  # "[REDACTED - not provided ...]"는 대괄호를 지우기 전에 알아본다
                dropped.append((raw_item, "placeholder"))
                continue
            item = strip_placeholders(META_PAREN.sub("", raw_item))
            if not item:
                dropped.append((raw_item, "placeholder"))
                continue
            ok, why = keep(item, note_nums)
            if ok and k == "prior_results" and same_day(item, note) and planned_test(item, note):
                ok, why = False, "leak"
            elif ok and k == "prior_results" and same_day(item, note):
                dropped.append((item, "same_day"))
                item = test_only(item)  # 결과는 빼고 "오늘 촬영함"만 남긴다
            elif ok and k == "medications" and new_prescription(item, dialogue):
                ok, why = False, "new_rx"
            elif ok and k == "vitals" and EXAM.search(item):
                ok, why = False, "exam"
            elif ok and k == "history" and ASSESSMENT.search(item) and not FAMILY.search(item):
                ok, why = False, "assessment"
            (out[k].append(item) if ok else dropped.append((item, why)))
    # 이름은 기록에 그대로 있어야 하고, 나이는 숫자가 기록에 있어야 한다
    name = strip_placeholders(chart.get("name", ""))
    out["name"] = name if name and name in note else ""
    age = strip_placeholders(chart.get("age", ""))
    # 나이 칸에 문장을 쓴 경우("...the note states she is 71 years old")가 있어 숫자만 남긴다
    age_num = re.search(r"\d+", age)
    out["age"] = age_num.group() if age_num and int(age_num.group()) in note_nums else ""
    # "F", "female", "Female"이 섞이지 않게 통일. 그 밖의 값(Non-binary 등)은 기록에 있을 때만 그대로 둔다
    sex = strip_placeholders(chart.get("sex", ""))
    out["sex"] = SEX.get(sex.lower(), sex if sex and sex.lower() in note.lower() else "")
    return out, dropped


def add_mentioned(chart, additions, note):
    """대화가 가리키는데 차트에 없던 항목을 더한다 (kaggle/09_complete_chart.py의 additions). (차트, 더한 항목, 뺀 항목)

    정답 기록에서 옮긴 값("note")은 처음 차트와 같은 규칙으로 거르고, 지어낸 값("invented")은 기록에 없는 것이 당연하므로
    숫자 확인만 빼고 거른다(자리 표시·답 누설은 똑같이 막는다).
    """
    chart = {k: list(v) if isinstance(v, list) else v for k, v in chart.items()}
    note_nums, added, dropped = numbers(note), [], []
    for a in additions or []:
        field, value = a.get("field"), strip_placeholders(a.get("value", ""))
        if field not in LIST_FIELDS or not value:
            continue
        ok, why = keep(value, note_nums if a.get("source") == "note" else numbers(value))
        if not ok:
            dropped.append((value, why))
        elif value.lower() not in (v.lower() for v in chart[field]):
            chart[field].append(value)
            added.append({"field": field, "value": value, "source": a.get("source"), "mentioned": a.get("mentioned")})
    return chart, added, dropped


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


def said_age(dialogue, age):
    """대화에서 환자 나이를 말했는가 ("i'm fifty years old", 의사의 소개 "karen is a 34 -year-old")."""
    intro = DIALOGUE_INTRO.search(dialogue.lower())
    return age in dialogue_ages(dialogue) or bool(intro and intro.group(3) == age)


def blank_age_example(cid, chart, dialogue, target_json):
    """나이 칸을 비우고 정답에서도 나이를 지운 짝 (chart_text, 정답 JSON 문자열). 만들 수 없으면 None.

    학습 차트에 나이가 늘 있으면 모델은 빈 나이 칸을 본 적이 없어, 빈칸을 보고도 나이를 지어낸다(차트 2차 스트레스
    테스트 15/29건). 같은 대화를 "나이 있음 → 씀", "빈칸 → 쓰지 않음" 두 가지로 보여 빈칸의 뜻을 가르친다.
    대화에서 나이를 말한 사례는 빈칸이어도 기록에 나이를 써야 맞으므로 만들지 않는다. 나머지 중 사례 id로 절반을 고른다.
    """
    age = chart["age"]
    if not age.isdigit() or zlib.crc32(cid.encode()) % 2 or said_age(dialogue, age):
        return None
    pattern = re.compile(rf"\b{age}(?:[- ]year[- ]old|\s?y\.\s?o\.|\s?y/o|[- ]yo)\b,? ?|\b(?:age|aged) {age}\b,? ?")
    target = {k: pattern.sub("", v) for k, v in json.loads(target_json).items()}
    target = {k: re.sub(r"\b([Aa])n (?=[b-df-hj-np-tv-z])", r"\1 ", v) for k, v in target.items()}  # "an 72-year-old woman"
    if any(re.search(rf"\b{age}\s?-?\s?(?:years?\b|y\.?\s?/?o\b)", v) for v in target.values()):  # 못 지운 나이 표기
        return None
    return render_chart(chart | {"age": ""}), json.dumps(target, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    # 프롬프트를 바꾸면 학습 데이터도 바뀐다. 이전 학습을 재현할 수 있게 새 폴더에 쓴다 (sft_chart = 최종 모델, sft_chart_v2 = 8차 값 규칙 실험 — 효과가 없어 프롬프트는 되돌렸고 재현은 커밋 e5846e4)
    ap.add_argument("--sft-dir", default="data/sft_chart")
    sft_dir = ROOT / ap.parse_args().sft_dir
    stats = {}
    sft_dir.mkdir(parents=True, exist_ok=True)
    for split, cases in sources().items():
        raw = {r["id"]: r["chart"] for r in load_jsonl(EXTRACTED / f"chart_{split}.jsonl")}
        comp_path = EXTRACTED / f"completion_{split}.jsonl"
        comp = {r["id"]: r for r in load_jsonl(comp_path)} if comp_path.exists() else {}
        reasons, kept, synthetic, rows = Counter(), Counter(), Counter(), []
        for cid, (dialogue, note, _) in cases.items():
            strict, dropped = clean_chart(raw.get(cid), note, dialogue)
            c = comp.get(cid) or {}
            demo = fill(cid, note, dialogue, strict)
            chart = strict | {k: demo[k] for k in ["name", "age", "sex"]}
            filled = [k for k in ["name", "age", "sex"] if demo[k] != strict[k]]
            chart, added, dropped_add = add_mentioned(chart, c.get("additions"), note)
            reasons.update(why for _, why in dropped + dropped_add)
            kept.update(k for k in LIST_FIELDS for _ in strict[k])
            kept.update(k for k in ["name", "age"] if strict[k])
            synthetic.update(filled)
            synthetic.update(f"{a['field']}({a['source']})" for a in added)
            rows.append({"id": cid, "chart": chart, "chart_text": render_chart(chart), "chart_strict": strict,
                         "chart_text_strict": render_chart(strict), "synthetic": filled, "added": added,
                         "demographics_source": demo["source"], "demographics_conflicts": demo["conflicts"],
                         "dropped": dropped + dropped_add})
        with open(ROOT / "data" / "chart" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {"cases": len(rows), "kept": dict(kept), "dropped": dict(reasons), "synthetic": dict(synthetic),
                        "empty_charts": sum(not any(r["chart"].values()) for r in rows)}

        if split in ("train", "valid"):
            with open(sft_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                blank = 0
                for r in rows:
                    dialogue, _, target_json = cases[r["id"]]
                    pairs = [(r["id"], r["chart_text"], target_json)]
                    extra = blank_age_example(r["id"], r["chart"], dialogue, target_json)
                    if extra:
                        pairs.append((r["id"] + "-blank-age", *extra))
                        blank += 1
                    for eid, chart_text, target in pairs:
                        messages = build_messages(dialogue, json_mode=True, chart_text=chart_text)
                        messages.append({"role": "assistant", "content": target})
                        f.write(json.dumps({"id": eid, "messages": messages}, ensure_ascii=False) + "\n")
                stats[split]["blank_age_examples"] = blank

    (ROOT / "results" / "chart_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
