"""생성 프롬프트. 기준선 측정과 SFT 학습 데이터가 같은 프롬프트를 써야
"학습 전 vs 학습 후" 비교가 프롬프트 차이 없이 모델 차이만 보여준다.

섹션 제목은 sectiontagger.py가 인식하는 표기(ACI-BENCH 정답 기록에 실제로 쓰인
대문자 제목)로만 지시한다 — 다른 표기를 쓰면 구역 분리가 틀어져 구역별 점수가
실제 품질과 무관하게 떨어진다.
"""
from schema import DIVISIONS

SYSTEM_PROMPT = """You are a medical scribe. Read the doctor-patient conversation and write the clinical note for this visit.

Rules:
- Only include information stated in the conversation. Do not invent findings, test results, or diagnoses.
- Write like a physician's clinical note: third person, concise, complete sentences.
- Use these uppercase section headings, each on its own line, in this order. Omit a section if the conversation has no information for it.
  CHIEF COMPLAINT
  HISTORY OF PRESENT ILLNESS
  REVIEW OF SYSTEMS
  PHYSICAL EXAMINATION
  RESULTS
  ASSESSMENT AND PLAN"""

JSON_INSTRUCTION = f"""

Return a JSON object with the keys {", ".join(DIVISIONS)}.
- subjective: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS and REVIEW OF SYSTEMS sections
- objective_exam: PHYSICAL EXAMINATION section
- objective_results: RESULTS section
- assessment_and_plan: ASSESSMENT AND PLAN section
Each value contains the section headings and text. Use an empty string when there is no information."""


CONVERSATION_RULE = "- Only include information stated in the conversation."
CHART_RULE = ("- Only include information stated in the conversation or the patient chart. "
              "Copy chart values exactly; never add a value that is in neither. "
              "An empty chart field means that information is not available.")
INVENT_RULE = "Do not invent findings, test results, or diagnoses."
# 7차: 원문에 '무엇'은 있고 '값'은 없을 때 전형적인 값을 채웠다("vitals look great" → 120/80, 약 이름만 → 흔한 용량).
# "숫자를 쓰지 마라"처럼 넓게 쓰지 않는다 — 학습 정답에 대화에 없는 날짜가 19개 있어 정답과 충돌한다.
# 예시 문장은 학습 정답이 실제로 쓰는 표현이다. 구체적인 정상값(120/80)은 떠올리게 할 수 있어 적지 않는다
VALUE_RULES = ("- If vital signs or exam findings are described only in words (\"vitals look good\", "
               "\"blood pressure is a little elevated\"), write the words, e.g. \"Vital signs are within normal limits.\" "
               "Do not fill in typical values.\n"
               "- If a lab result is described only as normal, high or low, write that description, not a value.\n"
               "- Write a medication dose only if it is stated. Otherwise write the medication without a dose.")

CHART_FIELDS = [("name", "Name"), ("age", "Age"), ("sex", "Sex"), ("vitals", "Vital signs"),
                ("prior_results", "Prior results"), ("history", "History"), ("medications", "Medications"),
                ("allergies", "Allergies")]


def render_chart(chart: dict) -> str:
    """차트 dict → 모델 입력용 텍스트. 모든 차트가 같은 항목을 같은 순서로 갖는다.

    모르는 값은 줄을 빼지 않고 이름만 남긴 채 비워 둔다("Age:"). 빈칸이 "없음"이라는 것은 학습 데이터의 짝으로 가르친다
    (build_chart_data.blank_age_examples: 같은 대화에 나이 칸을 비우고 정답에서도 나이를 지운 예).
    """
    lines = []
    for key, label in CHART_FIELDS:
        value = chart.get(key)
        value = "; ".join(value) if isinstance(value, list) else (value or "")
        lines.append(f"{label}: {value.strip()}".rstrip())
    return "\n".join(lines)


def build_messages(dialogue: str, json_mode: bool, chart_text: str | None = None) -> list:
    """chart_text가 있으면 실제 진료처럼 "차트 + 대화"를 입력으로 준다 (src/build_chart_data.py)."""
    system = SYSTEM_PROMPT + (JSON_INSTRUCTION if json_mode else "")
    if chart_text is None:
        user = dialogue
    else:
        system = system.replace(CONVERSATION_RULE, CHART_RULE).replace(INVENT_RULE, INVENT_RULE + "\n" + VALUE_RULES)
        user = f"PATIENT CHART\n{chart_text}\n\nCONVERSATION\n{dialogue}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
