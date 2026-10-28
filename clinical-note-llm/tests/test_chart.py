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
    assert dropped == [("all", "extract_failed")]
    assert render_chart(c).splitlines()[1] == "Age: not recorded" and render_chart(c).count("not recorded") == 8


def test_every_chart_lists_all_fields():
    text = render_chart(chart(age="50", medications=["lisinopril", "Lasix"]))
    assert text.splitlines() == ["Name: not recorded", "Age: 50", "Sex: not recorded", "Vital signs: not recorded",
                                 "Prior results: not recorded", "History: not recorded",
                                 "Medications: lisinopril; Lasix", "Allergies: not recorded"]


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


def test_labels_without_values_and_suspected_findings_are_dropped():
    c, dropped = clean_chart(chart(vitals=["Blood pressure", "Blood pressure 128/72"],
                                   history=["Past surgical history: None mentioned"],
                                   prior_results=["Urinary tract infection suspected"]), NOTE)
    assert c["vitals"] == ["Blood pressure 128/72"] and c["history"] == [] and c["prior_results"] == []
    assert sorted(why for _, why in dropped) == ["leak", "placeholder", "placeholder"]


def test_bracketed_and_note_referencing_placeholders_are_dropped():
    c, dropped = clean_chart(chart(vitals=["Height: [not specified]", "[not specified]"],
                                   medications=["Current medications and doses: Not specified in the note"]), NOTE)
    assert c["vitals"] == [] and c["medications"] == []
    assert [why for _, why in dropped] == ["placeholder"] * 3


def test_meta_comments_dropped_and_sex_normalized():
    c, dropped = clean_chart(chart(sex="F", history=["[REDACTED - not provided in the conversation or note]",
                                                     "Allergies not specified in the note.", "Hypertension"],
                                   prior_results=["Chest x-ray: normal (date not specified)"]), NOTE)
    assert c["sex"] == "Female" and c["history"] == ["Hypertension"]
    assert c["prior_results"] == ["Chest x-ray: normal"] and len(dropped) == 2


def test_fill_demographics_matches_context_and_is_repeatable():
    from build_chart_data import fill_demographics
    strict = chart(sex="", age="")
    a, filled = fill_demographics(strict, "D2N198", "", "He is a college student with back pain. His")
    assert filled == ["age", "sex"] and 18 <= int(a["age"]) <= 24 and a["sex"] == "Male"
    assert fill_demographics(strict, "D2N198", "", "He is a college student with back pain. His")[0]["age"] == a["age"]
    kept, filled = fill_demographics(chart(age="61", sex="Female"), "X", "", "")
    assert filled == [] and kept["age"] == "61"


def test_llm_demographics_fill_only_missing_slots():
    from build_chart_data import fill_demographics
    llm = {"name": "Sarah Thompson", "age": 34, "sex": "Female", "source": {"name": "stated", "age": "invented", "sex": "stated"}}
    c, filled = fill_demographics(chart(age="61"), "X", "", "", llm)
    assert filled == ["name", "sex"] and c["name"] == "Sarah Thompson" and c["age"] == "61" and c["sex"] == "Female"
    assert c["medications"] == []  # 인적사항 말고는 채우지 않는다


def test_bracket_placeholders_removed_everywhere():
    c, dropped = clean_chart(chart(vitals=["Blood Pressure: [value]", "Blood pressure 128/72"],
                                   history=["Past Medical History: Anxiety, Epilepsy; Past Surgical History: [details]"],
                                   medications=["Keppra [current dosage]", "[list]"],
                                   prior_results=["X-rays of the low back: Normal (Date: [date])"]), NOTE)
    assert c["vitals"] == ["Blood pressure 128/72"]
    assert c["history"] == ["Past Medical History: Anxiety, Epilepsy"]
    assert c["medications"] == ["Keppra"] and c["prior_results"] == ["X-rays of the low back: Normal"]
    assert "[" not in render_chart(c)


def test_age_sentence_reduced_to_number_and_bare_lab_label_dropped():
    c, _ = clean_chart(chart(age="While the exact age is not provided, the note states she is 50 years old",
                             prior_results=["Hemoglobin A1c:", "BNP 3000 in 03/2021"]), NOTE)
    assert c["age"] == "50" and c["prior_results"] == ["BNP 3000 in 03/2021"]
