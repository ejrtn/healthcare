"""근거 검사 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from faithfulness import check  # noqa: E402


def test_spoken_number_words_count_as_support():
    # D2N117: 대화에서 "one thousand milligrams"라고 말한 용량을 근거 없음으로 잘못 표시하던 버그
    dialogue = "[doctor] i'm going to increase your metformin to one thousand milligrams twice per day ."
    assert check("Increase metformin to 1000 mg twice daily.", dialogue)["unsupported_numbers"] == []
    # D2N164: 용량을 말하지 않았는데 기록에 쓰면 표시한다
    dialogue = "[doctor] i'm gon na go ahead and start you on some hydrochlorothiazide , it's a pretty low dose ."
    assert check("Start hydrochlorothiazide 12.5 mg daily.", dialogue)["unsupported_numbers"] == ["12.5"]


def test_checker_v2_reads_spoken_number_forms():
    # 표시된 51문장 중 검사기 오탐 22개의 원인이던 표현들
    dialogue = ("[patient] about two and a half weeks ago . [doctor] it's 1,792 grams . terminal extension to a hundred "
                "and fifteen degrees . attached three hundred and sixty degrees . one seventy five over ninety two . "
                "i had to call nine one one .")
    note = "Began 2.5 weeks ago. Resection 1792 g. Extension to 115 degrees. Attached 360 degrees. BP 175/92. Called 911."
    assert check(note, dialogue)["unsupported_numbers"] == []


def test_scale_denominators_are_convention_but_numerators_are_checked():
    dialogue = "[patient] it's around a seven . [doctor] i do hear a slight murmur ."
    r = check("Pain is 8/10. Grade 3/6 systolic murmur. Pain was 7 out of 10.", dialogue)
    assert r["unsupported_numbers"] == ["8", "3"]  # 분모 10·6은 세지 않고, 분자는 1~3이어도 확인한다


def test_decades_and_compound_tens():
    dialogue = "[doctor] oxygen is about ninety four percent . pressure is one forty four over seventy two ."
    assert check("Readings in the lower 90s.", dialogue)["unsupported_numbers"] == []
    # "one forty four"은 144다. 예전 검사기는 "forty"만 떼어 140·40도 근거로 봤다
    assert check("Blood pressure 140/72. She is a 40-year-old female.", dialogue)["unsupported_numbers"] == ["140"]
    assert check("She is a 40-year-old female.", dialogue)["invented_age"] == ["40"]


def test_ordinals_and_years():
    # API 끝까지 시험(D2N108·D2N138)에서 찾은 표현: 서수는 기수로, "twenty twenty one"은 연도로 읽는다
    dialogue = "[doctor] just proximal to the fifth mtp joint . refills until october thirty first , twenty twenty one ."
    assert check("Proximal to the 5th MTP joint. Refills until October 31st, 2021.", dialogue)["unsupported_numbers"] == []
    assert check("Refills until October 31st, 2021.", "[doctor] refills are available .")["unsupported_numbers"] == ["31", "2021"]
