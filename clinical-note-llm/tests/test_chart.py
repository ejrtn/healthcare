"""차트 걸러내기·입력 만들기 테스트."""
import json
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
    assert render_chart(c).splitlines()[1] == "Age:" and all(line.endswith(":") for line in render_chart(c).splitlines())


def test_every_chart_lists_all_fields():
    text = render_chart(chart(age="50", medications=["lisinopril", "Lasix"]))
    assert text.splitlines() == ["Name:", "Age: 50", "Sex:", "Vital signs:", "Prior results:", "History:",
                                 "Medications: lisinopril; Lasix", "Allergies:"]


def test_same_day_results_are_dropped_but_outside_studies_kept():
    note = ("X-rays of the right knee were obtained and reviewed in office today. "
            "CT scan of chest, obtained at an outside facility, is reviewed today and demonstrates a 2 cm nodule. "
            "Martha Lopez presents to the clinic today. Hemoglobin A1c was 8 last month.")
    c, dropped = clean_chart(chart(prior_results=["X-rays of the right knee: no fracture",
                                                  "CT scan of chest: 2 cm nodule", "Hemoglobin A1c 8"]), note)
    assert c["prior_results"] == ["CT scan of chest: 2 cm nodule", "Hemoglobin A1c 8"]
    assert dropped == [("X-rays of the right knee: no fracture", "same_day")]


def test_blank_age_pair_removes_age_from_answer():
    from build_chart_data import blank_age_example
    target = json.dumps({"subjective": "Martha Collins is an 72-year-old female with 50 mg lisinopril."})
    cid = next(c for c in (f"D2N{i:03d}" for i in range(200)) if blank_age_example(c, chart(age="72"), "", target))
    chart_text, answer = blank_age_example(cid, chart(age="72"), "", target)
    assert chart_text.splitlines()[1] == "Age:"
    assert json.loads(answer)["subjective"] == "Martha Collins is a female with 50 mg lisinopril."
    # 대화에서 나이를 말했으면 빈칸이어도 기록에 써야 하므로 짝을 만들지 않는다
    assert blank_age_example(cid, chart(age="72"), "[patient] i'm 72 years old", target) is None


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


def test_add_mentioned_items_note_values_checked_invented_allowed():
    from build_chart_data import add_mentioned
    additions = [
        {"field": "vitals", "value": "Blood pressure 128/72", "source": "note", "mentioned": "your blood pressure"},
        {"field": "vitals", "value": "Heart rate 88", "source": "note", "mentioned": "heart rate"},  # 기록에 없는데 note라고 함
        {"field": "medications", "value": "lisinopril 20 mg daily", "source": "invented", "mentioned": "the lisinopril"},
        {"field": "medications", "value": "start metformin 1000 mg", "source": "invented", "mentioned": "x"},  # 계획
        {"field": "allergies", "value": "[details]", "source": "invented", "mentioned": "x"},
    ]
    c, added, dropped = add_mentioned(chart(), additions, NOTE)
    assert c["vitals"] == ["Blood pressure 128/72"] and c["medications"] == ["lisinopril 20 mg daily"]
    assert [a["source"] for a in added] == ["note", "invented"]
    assert sorted(why for _, why in dropped) == ["leak", "not_in_note"]


def test_note_reference_without_article_dropped():
    c, _ = clean_chart(chart(allergies=["Unknown from note provided."],
                             vitals=["Blood glucose: Not specified in note", "Blood pressure 128/72"],
                             history=["Older sister has an unknown heart problem"]), NOTE)
    assert c["allergies"] == [] and c["vitals"] == ["Blood pressure 128/72"]
    assert c["history"] == ["Older sister has an unknown heart problem"]  # 실제 가족력 정보는 남긴다
