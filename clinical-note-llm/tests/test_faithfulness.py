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
