"""진료기록 ↔ 4구역 JSON 변환.

구역 나누기는 ACI-BENCH 공식 평가(evaluation/evaluate_fullnote.py)가 쓰는
SectionTagger를 그대로 쓴다(sectiontagger.py). 정답 데이터 변환과 채점이 같은
규칙으로 구역을 나눠야, 모델이 낸 JSON을 진료기록으로 되돌렸을 때 공식 점수와
바로 비교된다.
"""
from sectiontagger import SectionTagger

DIVISIONS = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]

NOTE_JSON_SCHEMA = {
    "type": "object",
    "properties": {d: {"type": "string"} for d in DIVISIONS},
    "required": DIVISIONS,
    "additionalProperties": False,
}

_tagger = SectionTagger()


def split_note(note: str) -> dict:
    """진료기록 텍스트를 4구역으로 나눈다. 없는 구역은 빈 문자열."""
    divisions = dict.fromkeys(DIVISIONS, "")
    for label, _, _, start, _, end in _tagger.divide_note_by_metasections(note):
        divisions[label] = note[start:end].strip()
    return divisions


def render_note(divisions: dict) -> str:
    """4구역 JSON을 다시 진료기록 텍스트로 합친다 (구역 순서 고정)."""
    return "\n\n".join(divisions[d].strip() for d in DIVISIONS if divisions.get(d, "").strip())
