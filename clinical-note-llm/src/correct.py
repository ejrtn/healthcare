"""규칙 보정: 근거 검사(faithfulness.check)가 입력(차트 + 대화)에 없다고 한 값을 기록에서 뺀다. LLM은 쓰지 않는다.

  "Uric acid was elevated at 7 mg/dL."                 → "Uric acid was elevated."
  "I have prescribed hydrochlorothiazide 12.5 mg daily" → "I have prescribed hydrochlorothiazide daily"
  "Jacqueline is a pleasant 45-year-old female"         → "Jacqueline is a pleasant female"

값을 고쳐 넣지는 않는다(예: 208/92 → 200/90). 무엇이 맞는 값인지 고르는 것은 규칙이 할 수 없는 판단이라, 빼고
의료진이 채우게 한다. 문장 전체를 지우지 않아 맞는 정보(약 이름, "elevated" 같은 판단)는 남는다.
값 앞뒤의 말("at", "of", 단위, "twice daily")을 함께 지워 문장이 어색해지는 것을 줄인다. 뺐는데도 근거 없는 값이 남으면
(규칙이 못 지운 경우) 고치지 않고 그대로 두어 확인 표시가 유지되게 한다. 검사 이름(CT 등)은 빼면 문장이 깨지므로 표시만 한다.
"""
import re

from faithfulness import check

PREFIX = r"(?:(?:at|of|to|from|around|about|approximately|is|was|were|be|to be|as an?|in the (?:last|past)|every|by|for|up to)\s+)?"
UNIT = (r"(?:\s*(?:%|percent|mg/dl|mg|mcg|micrograms?|milligrams?|grams?|g|kg|lbs?|pounds?|ml|units?|mm/hr|mmhg|"
        r"beats per minute|bpm|degrees?|hours?|days?|weeks?|months?|years?|minutes?|cm|mm|times?))?")
AFTER = r"(?:\s+(?:predicted|(?:once|twice|three times|four times)\s+(?:daily|a day|per day)|daily|a day|per day|at night))?"


def value_pattern(value: str) -> re.Pattern:
    """값 하나를 지울 범위: 앞말 + 값(범위·분수·척도 포함) + 단위 + 빈도."""
    v = re.escape(value)
    number = rf"(?:grade\s+)?(?:\d+(?:\.\d+)?\s*[-/]\s*)?{v}(?:\s*(?:[-/]|out of)\s*\d+(?:\.\d+)?)?"
    return re.compile(rf"\s*\b{PREFIX}{number}(?![\d.]\d){UNIT}{AFTER}\b", re.I)


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r",\s*([.;])", r"\1", text)
    text = re.sub(r",?\s*\b(?:especially|including)(?=[.;])", "", text)  # 값을 빼고 남은 꼬리말
    text = re.sub(r"\s*\(\s*[,;/]?\s*\)", "", text)  # 값만 들어 있던 괄호 "Normal (70-99 mg/dL)" → "Normal"
    text = re.sub(r"\(\s*[,;]\s*", "(", text)
    text = re.sub(r"(^|[.!?]\s+|\n[-•]?\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def correct_text(text: str, source: str) -> tuple[str, list[dict]]:
    """한 구역의 글을 보정한다. 반환: 보정한 글, 바꾼 내역 [{"value", "removed", "kind"}]."""
    r = check(text, source)
    edits = []
    for age in dict.fromkeys(r["invented_age"]):
        new = re.sub(rf"\b(a|an)\s+{age}[- ]year[- ]old\b", "a", text, flags=re.I)
        new = re.sub(rf"\b{age}[- ]year[- ]old\s*", "", new, flags=re.I)
        if new != text:
            edits.append({"value": f"{age}-year-old", "removed": f"{age}-year-old", "kind": "age"})
            text = new
    for value in dict.fromkeys(r["unsupported_numbers"]):
        m = value_pattern(value).search(text)
        if m:
            edits.append({"value": value, "removed": m.group().strip(), "kind": "number"})
            text = text[:m.start()] + text[m.end():]
    return tidy(text), edits


def correct_note(divisions: dict, source: str) -> tuple[dict, list[dict], dict]:
    """4구역 기록 전체를 보정한다. 반환: 보정한 기록, 바꾼 내역, 남은 확인 항목(규칙이 못 지운 값·검사 이름)."""
    out, edits = {}, []
    for key, text in divisions.items():
        new, e = correct_text(text, source)
        out[key] = new
        edits += [x | {"division": key} for x in e]
    left = check("\n\n".join(out.values()), source)
    return out, edits, {k: left[k] for k in ("unsupported_numbers", "unsupported_tests", "invented_age")}
