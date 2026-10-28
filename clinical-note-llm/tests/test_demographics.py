"""이름·나이·성별 규칙(src/demographics.py) 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from demographics import fill  # noqa: E402


def test_note_intro_wins_over_family_age():
    note = "Martha Collins is a pleasant 50-year-old female. Her 7-year-old son is well."
    r = fill("X", note, "[doctor] hi , martha . how are you ?")
    assert (r["name"], r["age"], r["sex"]) == ("Martha Collins", "50", "Female")
    assert r["source"]["age"] == "note:intro" and not r["conflicts"]


def test_age_formats_and_event_ages():
    assert fill("X", "Sophia Brown is a 75 y.o. female.", "")["age"] == "75"
    assert fill("X", "Katherine Lopez is a 22-month-old female. Brother: 9 years old.", "")["age"] == "22 months"
    # "since age 12"는 증상이 시작된 때, "when i was around eight years old"는 지난 일
    r = fill("X", "She has had reflux since approximately age 12. Her mother had cancer.",
             "[patient] when i was around eight years old i was hospitalized")
    assert r["source"]["age"] == "random"


def test_greeting_needs_separator_and_real_name():
    # "hirsutism"의 hi, "hi, how are you"의 how는 이름이 아니다
    r = fill("X", "The patient has hirsutism. She is doing well. Her symptoms improved.",
             "[patient] fine . [doctor] hi, how are you ? [patient] fine . i have hirsutism")
    assert r["source"]["name"].startswith("random") and r["sex"] == "Female"


def test_greeting_name_with_title_surname():
    r = fill("X", "Ms. Lopez reports knee pain. She denies fever. Her knee is swollen.",
             "[doctor] hi sarah , good to see you . sarah , tell me about the knee")
    assert r["name"] == "Sarah Lopez" and r["source"]["name"] == "dialogue:greeting+surname:note:title"


def test_title_surname_is_not_used_as_first_name():
    # "hey , ms. james"의 james는 성이다 ("James James"가 되었다)
    r = fill("X", "Ms. James is a 42-year-old female. She fell.", "[doctor] hey , ms. james . nice to meet you .")
    assert r["name"].endswith(" James") and not r["name"].startswith("James")


def test_dialogue_intro():
    r = fill("X", "Patient presents with eye tics. Her father provided a video.",
             "[doctor] karen nelson is a 3 -year-old female with no significant past medical history")
    assert (r["name"], r["age"]) == ("Karen Nelson", "3") and r["source"]["age"] == "dialogue:intro"


def test_sex_special_cases():
    r = fill("X", "They note that their parents smoked.", "[doctor] julie cruz is a 17 -year-old non- binary minor")
    assert r["sex"] == "Non-binary"
    # 정답 기록의 오타: "male"인데 she/her만 쓴다 → 대명사를 따르고 경고를 남긴다
    note = "Michelle King is a 44-year-old male. " + "She skis. Her knee hurts. " * 3
    r = fill("X", note, "")
    assert r["sex"] == "Female" and r["conflicts"]


def test_random_values_are_repeatable():
    a = fill("D2N999", "Patient presents for follow up. Patient is a student.", "")
    b = fill("D2N999", "Patient presents for follow up. Patient is a student.", "")
    assert a == b and 18 <= int(a["age"]) <= 24 and a["source"]["age"] == "random"
