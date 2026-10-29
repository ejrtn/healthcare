"""규칙 보정 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from correct import correct_note, correct_text  # noqa: E402

SOURCE = ("[doctor] your uric acid came back . i'll start you on some hydrochlorothiazide , a pretty low dose . "
          "take ibuprofen up to three times a day . i'll increase your metformin to one thousand milligrams twice a day .")


def test_removes_unsupported_values_with_their_words():
    cases = {
        "Uric acid was elevated at 7 mg/dL.": "Uric acid was elevated.",
        "I have prescribed hydrochlorothiazide 12.5 mg daily for hypertension.":
            "I have prescribed hydrochlorothiazide for hypertension.",
        "Continue ibuprofen every 6-8 hours as needed, up to 3 times per day.":
            "Continue ibuprofen as needed, up to 3 times per day.",
        "Increase metformin from 500 mg twice daily to 1000 mg twice daily.":
            "Increase metformin to 1000 mg twice daily.",  # 대화에 있는 1000은 남긴다
    }
    for before, after in cases.items():
        assert correct_text(before, SOURCE)[0] == after


def test_invented_age_removed_and_supported_text_untouched():
    assert correct_text("Martha is a pleasant 45-year-old female.", SOURCE)[0] == "Martha is a pleasant female."
    text = "Start hydrochlorothiazide. Increase metformin to 1000 mg twice daily."
    assert correct_text(text, SOURCE) == (text, [])


def test_correct_note_reports_what_rules_cannot_fix():
    note = {"subjective": "Knee pain.", "objective_results": "CT of the knee shows a tear. Uric acid 7 mg/dL."}
    fixed, edits, left = correct_note(note, SOURCE)
    assert fixed["objective_results"] == "CT of the knee shows a tear. Uric acid."
    assert [e["value"] for e in edits] == ["7"] and left["unsupported_tests"] == ["ct"]  # 검사 이름은 표시만


def test_empty_parentheses_left_by_removed_values_are_dropped():
    text, _ = correct_text("Glucose: Normal (70-99 mg/dL). Cholesterol: Elevated (210 mg/dL).",
                           "[doctor] your cholesterol was elevated at two ten .")
    assert text == "Glucose: Normal. Cholesterol: Elevated (210 mg/dL)."


def test_repeated_values_ordinals_and_mm_hg():
    # API 끝까지 시험에서 남았던 경우: 같은 값이 여러 번(D2N138), 서수 "31st", "mm Hg"의 Hg만 남던 것(D2N161)
    source = "[doctor] your blood pressure was elevated in the emergency room . refills are available ."
    text, edits = correct_text("Refills until October 31st, 2021. Inhaler refills until October 31st, 2021.", source)
    assert text == "Refills until October. Inhaler refills until October." and len(edits) == 4
    text, _ = correct_text("Blood pressure was elevated at 208/72 mm Hg.", source)
    assert text == "Blood pressure was elevated."


def test_value_inside_decimal_is_not_removed():
    text, _ = correct_text("Result 7.6. Follow up for 6 weeks.", "[doctor] the result was 76 .")
    assert text == "Result 7.6. Follow up."  # 근거 없는 6만 빼고 7.6 속의 6은 둔다
