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
              "Copy chart values exactly; never add a value that is in neither.")

CHART_FIELDS = [("name", "Name"), ("age", "Age"), ("sex", "Sex"), ("vitals", "Vital signs"),
                ("prior_results", "Prior results"), ("history", "History"), ("medications", "Medications"),
                ("allergies", "Allergies")]


def render_chart(chart: dict) -> str:
    """차트 dict → 모델 입력용 텍스트. 빈 항목은 줄을 만들지 않는다."""
    lines = []
    for key, label in CHART_FIELDS:
        value = chart.get(key)
        value = "; ".join(value) if isinstance(value, list) else (value or "")
        if value.strip():
            lines.append(f"{label}: {value.strip()}")
    return "\n".join(lines) or "(no chart information)"


def build_messages(dialogue: str, json_mode: bool, chart_text: str | None = None) -> list:
    """chart_text가 있으면 실제 진료처럼 "차트 + 대화"를 입력으로 준다 (src/build_chart_data.py)."""
    system = SYSTEM_PROMPT + (JSON_INSTRUCTION if json_mode else "")
    if chart_text is None:
        user = dialogue
    else:
        system = system.replace(CONVERSATION_RULE, CHART_RULE)
        user = f"PATIENT CHART\n{chart_text}\n\nCONVERSATION\n{dialogue}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
