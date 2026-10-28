"""이름·나이·성별 규칙(src/demographics.py) 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from demographics import fill  # noqa: E402


def test_note_intro_wins_and_family_age_is_flagged():
    note = "Martha Collins is a pleasant 50-year-old female. Her 7-year-old son is well."
    r = fill("X", note, "[doctor] hi , martha . how are you ?")
    assert (r["name"], r["age"], r["sex"]) == ("Martha Collins", "50", "Female")
    assert r["source"]["age"] == "note:intro" and any("여럿" in c for c in r["conflicts"])


def test_greeting_needs_separator_and_real_name():
    # "hirsutism"의 hi, "hi, how are you"의 how는 이름이 아니다
    r = fill("X", "The patient has hirsutism. She is doing well. Her symptoms improved.",
             "[doctor] hi, how are you ? [patient] fine . i have hirsutism")
    assert r["source"]["name"].startswith("random") and r["sex"] == "Female"


def test_greeting_name_with_title_surname():
    r = fill("X", "Ms. Lopez reports knee pain. She denies fever. Her knee is swollen.",
             "[doctor] hi sarah , good to see you . sarah , tell me about the knee")
    assert r["name"] == "Sarah Lopez" and r["source"]["name"] == "dialogue:greeting+surname:note:title"


def test_random_values_are_repeatable():
    a = fill("D2N999", "Patient presents for follow up. Patient is a student.", "")
    b = fill("D2N999", "Patient presents for follow up. Patient is a student.", "")
    assert a == b and 18 <= int(a["age"]) <= 24 and a["source"]["age"] == "random"
