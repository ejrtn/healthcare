"""DPO 짝 고르기·판정 결과 처리 테스트 (GPU 없이 돈다)."""
import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle"))

from build_dpo_pairs import pick_pairs, score  # noqa: E402

judge = importlib.import_module("06_judge")

ARGS = Namespace(min_gap=2, max_pairs=2, rouge_slack=0.03, number_slack=1, length_ratio=0.85)
DIALOGUE = ("[doctor] any loss of control of your bladder or bowels ? [patient] yeah i pooped myself once . "
            "[doctor] your blood pressure is one twenty over eighty .")


def note(subjective, exam="PHYSICAL EXAMINATION\n\nBlood pressure 120/80."):
    return json.dumps({"subjective": subjective, "objective_exam": exam,
                       "objective_results": "", "assessment_and_plan": ""})


def cand(sample, subjective, **kw):
    return {"sample": sample, "prediction": note(subjective, **kw), "finish_reason": "stop"}


def contradiction():
    return {"type": "contradiction", "note_text": "denies incontinence", "evidence": "pooped myself once"}


GOOD = "HISTORY OF PRESENT ILLNESS\n\nOne episode of fecal incontinence."
BAD = "HISTORY OF PRESENT ILLNESS\n\nThe patient denies bowel incontinence."


def test_score_counts_judge_and_rule_flags():
    s = score(cand(0, BAD, exam="PHYSICAL EXAMINATION\n\nBlood pressure 150/80."), [contradiction()], DIALOGUE, BAD)
    assert s["contradictions"] == 1 and s["rule_flags"] == 1  # 150은 대화에 없다
    assert s["penalty"] == 3 and s["usable"]


def test_unusable_candidates_are_skipped():
    truncated = cand(1, GOOD) | {"finish_reason": "length"}
    broken = {"sample": 2, "prediction": "{not json", "finish_reason": "stop"}
    assert not score(truncated, [], DIALOGUE, GOOD)["usable"]
    assert not score(broken, [], DIALOGUE, GOOD)["usable"]
    assert not score(cand(3, GOOD), None, DIALOGUE, GOOD)["usable"]  # 판정 실패


def test_pair_prefers_faithful_answer():
    scored = [score(cand(0, GOOD), [], DIALOGUE, GOOD),
              score(cand(1, BAD), [contradiction()], DIALOGUE, GOOD)]
    [(chosen, rejected)] = pick_pairs(scored, ARGS)
    assert chosen["sample"] == 0 and rejected["sample"] == 1


def test_shorter_answer_without_numbers_is_not_chosen():
    """숫자를 아예 빼서 벌점이 0인 답은 좋은 답이 될 수 없다 (꼼수 막기)."""
    empty = score(cand(0, "HISTORY OF PRESENT ILLNESS\n\nSeen today.", exam=""), [], DIALOGUE, GOOD)
    bad = score(cand(1, BAD), [contradiction()], DIALOGUE, GOOD)
    assert empty["penalty"] == 0
    assert pick_pairs([empty, bad], ARGS) == []


def test_small_penalty_gap_makes_no_pair():
    a = score(cand(0, GOOD), [], DIALOGUE, GOOD)
    b = score(cand(1, GOOD), [{"type": "unsupported", "note_text": "x", "evidence": "not mentioned"}], DIALOGUE, GOOD)
    assert pick_pairs([a, b], ARGS) == []


def test_parse_issues():
    ok = json.dumps({"issues": [contradiction(), {"type": "other", "note_text": "", "evidence": ""}]})
    assert judge.parse_issues(ok) == [contradiction()]
    assert judge.parse_issues('{"issues": [{"type": "contra') is None  # 판정 출력이 잘림
    s = judge.summarize([{"issues": [contradiction()]}, {"issues": []}, {"issues": None}])
    assert s["judge_failed"] == 1 and s["notes_with_contradiction"] == 1 and s["contradictions_per_note"] == 0.5
