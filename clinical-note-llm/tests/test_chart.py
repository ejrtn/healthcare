"""차트 걸러내기·입력 만들기 테스트."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from build_chart_data import clean_chart  # noqa: E402
from prompts import CHART_RULE, build_messages, render_chart  # noqa: E402

NOTE = ("Martha Lopez is a 50-year-old female. Blood pressure 128/72. BNP 3000 in 03/2021. "
        "She takes metformin 500 mg twice daily. Plan: increase metformin to 1000 mg.")
DIALOGUE = "[doctor] hi martha . are you taking the metformin five hundred ? [patient] yes ."


def chart(**kw):
    base = {"name": "", "age": "", "sex": "", "vitals": [], "prior_results": [], "history": [],
            "medications": [], "allergies": []}
    return base | kw


def test_keeps_chart_facts_absent_from_dialogue():
    c, dropped = clean_chart(chart(name="Martha Lopez", age="50", vitals=["Blood pressure 128/72"],
                                   prior_results=["BNP 3000 in 03/2021"]), NOTE)
    assert c["name"] == "Martha Lopez" and c["age"] == "50"
    assert c["vitals"] == ["Blood pressure 128/72"] and c["prior_results"] == ["BNP 3000 in 03/2021"]
    assert dropped == []


def test_drops_invented_and_leaked_items_but_keeps_spoken_ones():
    c, dropped = clean_chart(chart(vitals=["Heart rate 88"],  # 기록에 없는 값 → 추출 모델이 지어냄
                                   medications=["metformin 500 mg twice daily",  # 대화에도 나오지만 차트 정보라 남긴다
                                                "metformin 1000 mg (increase planned)"]),  # 이번 진료의 계획
                             NOTE)
    assert c["vitals"] == [] and c["medications"] == ["metformin 500 mg twice daily"]
    assert sorted(why for _, why in dropped) == ["leak", "not_in_note"]


def test_failed_extraction_gives_empty_chart():
    c, dropped = clean_chart(None, NOTE)
    assert render_chart(c) == "(no chart information)" and dropped == [("all", "extract_failed")]


def test_chart_prompt():
    plain = build_messages(DIALOGUE, json_mode=True)
    with_chart = build_messages(DIALOGUE, json_mode=True, chart_text="Age: 50")
    assert plain[1]["content"] == DIALOGUE and CHART_RULE not in plain[0]["content"]
    assert CHART_RULE in with_chart[0]["content"]
    assert with_chart[1]["content"].startswith("PATIENT CHART\nAge: 50\n\nCONVERSATION\n")


def test_placeholder_values_are_dropped():
    c, dropped = clean_chart(chart(sex="N/A", allergies=["none mentioned", "No known allergies"]), NOTE)
    assert c["sex"] == "" and c["allergies"] == ["No known allergies"]
    assert dropped == [("none mentioned", "placeholder")]
