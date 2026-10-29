"""근거 확인 에이전트: 근거 검사가 표시한 문장을 LLM이 도구로 근거를 찾아 유지·수정·삭제한다.

흐름 (문장 하나마다)
  1. 근거 검사(faithfulness.check)가 입력에 없는 숫자·검사·나이를 표시 → 그 값이 든 문장을 고른다.
  2. LLM에 문장과 도구를 준다. LLM이 검색어를 정해 도구를 부르고(Tool Calling), 우리 코드가 실행해 결과를 돌려준다.
     - search_dialogue(query): 이번 대화에서 관련 발화 검색 (evidence.py, BM25)
     - get_chart(field): 진료 전 차트의 한 항목
  3. 도구 호출이 끝나면 판단을 JSON 스키마로 강제해 받는다(Structured Output) — keep / fix / remove + 근거 발화 번호.
  4. 안전장치: 고친 문장에도 근거 없는 값이 남거나, 형식이 깨지거나, 도구 호출이 MAX_ROUNDS를 넘으면 원문을 그대로 둔다.
     고치지 못해도 지금(표시만)보다 나빠지지 않게 한다.

도구 인자는 하나씩만 받는다 — 3B 모델이 인자 여러 개짜리 도구 호출에서 값을 섞는 것을 이전 프로젝트에서 확인했다.
"""
import json
import re

from evidence import search, split_turns
from faithfulness import check

MAX_ROUNDS = 3
CHART_FIELDS = ["vitals", "prior_results", "history", "medications", "allergies"]

TOOLS = [
    {"type": "function", "function": {
        "name": "search_dialogue",
        "description": "Search this visit's conversation for the turns about a key term. Use the drug, test, "
                       "measurement or event the flagged value belongs to (e.g. 'hydrochlorothiazide', 'ESR', "
                       "'follow up'), not the number itself.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_chart",
        "description": "Read one field of the patient's pre-visit chart.",
        "parameters": {"type": "object", "properties": {"field": {"type": "string", "enum": CHART_FIELDS}},
                       "required": ["field"]}}},
]

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["keep", "fix", "remove"]},
        "new_text": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["verdict", "new_text", "evidence"],
}

SYSTEM = """You verify one sentence of a clinical note against this visit's evidence: the patient chart and the doctor-patient conversation.
A rule flagged values in the sentence that do not appear literally in the evidence.
Use the tools to find the evidence. Search the conversation with the term the value belongs to, not with the number.
Then decide:
- keep: the evidence supports the sentence (for example the value was said in words, like "one thousand").
- fix: the evidence supports the fact but not the flagged value, or gives a different value. Rewrite the sentence using only the evidence; drop or correct the unsupported value, e.g. "dose not stated".
- remove: nothing in the evidence supports the sentence.
Never add information that is not in the evidence."""

FINAL = ('Give your decision as JSON: {"verdict": "keep" | "fix" | "remove", "new_text": the rewritten sentence '
         '(the original sentence for keep, empty for remove), "evidence": [turn numbers you relied on]}.')

SENT = re.compile(r"(?:[^.\n]|\.(?=\d))+(?:\.|\n|$)")  # 소수점(12.5)은 문장 끝이 아니다


def flagged_sentences(note: str, flags: dict) -> list[dict]:
    """근거 검사 결과 → 표시된 값이 든 문장들 [{"sentence", "values"}]. 같은 문장의 값은 한 번에 다룬다."""
    values = flags["unsupported_numbers"] + flags["unsupported_tests"] + [f"{a}-year-old" for a in flags["invented_age"]]
    items = {}
    for v in dict.fromkeys(values):
        pat = re.compile(rf"(?<![\w.]){re.escape(v)}(?![\w])", re.I)
        for s in SENT.findall(note):
            if pat.search(s):
                items.setdefault(s.strip(), []).append(v)
    return [{"sentence": s, "values": vs} for s, vs in items.items()]


def run_tool(name: str, args: dict, turns: list[dict], chart: dict, method: str) -> str:
    if name == "search_dialogue":
        hits = search(str(args.get("query", "")), turns, method=method)
        return "\n".join(f"#{t['turn']} {t['text']}" for t in hits) or "(no matching turns)"
    if name == "get_chart":
        field = args.get("field")
        if field not in CHART_FIELDS:
            return f"unknown field; choose one of {CHART_FIELDS}"
        value = chart.get(field) or []
        return "; ".join(value) if isinstance(value, list) else str(value) or "(empty)"
    return f"unknown tool {name}"


async def verify_sentence(client, model: str, item: dict, turns: list[dict], chart: dict, source: str,
                          method: str = "bm25") -> dict:
    """문장 하나를 확인한다. 반환: verdict, new_text, evidence, tool_calls(기록), applied(실제 반영 여부), reason."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Flagged values: {', '.join(item['values'])}\nSentence: {item['sentence']}"}]
    calls = []
    for _ in range(MAX_ROUNDS):
        resp = await client.chat.completions.create(model=model, messages=messages, tools=TOOLS,
                                                    tool_choice="auto", temperature=0, max_tokens=256)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(tc.function.name, args, turns, chart, method)
            calls.append({"tool": tc.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    else:
        return {**item, "verdict": "keep", "new_text": item["sentence"], "evidence": [], "tool_calls": calls,
                "applied": False, "reason": "max_rounds"}

    messages.append({"role": "user", "content": FINAL})
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=256,
        response_format={"type": "json_schema", "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}})
    try:
        out = json.loads(resp.choices[0].message.content or "")
        verdict, new_text, evidence = out["verdict"], str(out.get("new_text", "")).strip(), list(out.get("evidence", []))
    except (json.JSONDecodeError, KeyError, TypeError):
        return {**item, "verdict": "keep", "new_text": item["sentence"], "evidence": [], "tool_calls": calls,
                "applied": False, "reason": "invalid_json"}

    result = {**item, "verdict": verdict, "new_text": new_text, "evidence": evidence, "tool_calls": calls}
    if verdict == "fix":
        left = check(new_text, source)
        if not new_text or any(left[k] for k in ("unsupported_numbers", "unsupported_tests", "invented_age")):
            return result | {"applied": False, "reason": "fix_still_unsupported"}
    return result | {"applied": verdict in ("fix", "remove"), "reason": ""}


def apply(note: str, results: list[dict]) -> str:
    for r in results:
        if r["applied"]:
            note = note.replace(r["sentence"], r["new_text"] if r["verdict"] == "fix" else "", 1)
    note = re.sub(r"[ \t]{2,}", " ", note)
    return re.sub(r"(^|\n)[ \t]+", r"\1", note)  # 지운 문장 자리에 남은 공백 정리


def apply_divisions(divisions: dict, results: list[dict]) -> dict:
    """4구역 JSON 기록에 반영한다. 문장은 한 구역 안에 있으므로 그 구역에서만 바꾼다."""
    out = dict(divisions)
    for r in results:
        if not r["applied"]:
            continue
        for k, text in out.items():
            if r["sentence"] in text:
                out[k] = apply(text, [r])
                break
    return out


async def revise(client, model: str, note: str, dialogue: str, chart: dict | None = None,
                 chart_text: str = "", method: str = "bm25") -> tuple[str, list[dict]]:
    """기록 전체: 표시된 문장마다 확인하고 반영한 기록과 문장별 기록을 돌려준다."""
    source = (chart_text + "\n\n" + dialogue) if chart_text else dialogue
    items = flagged_sentences(note, check(note, source))
    turns = split_turns(dialogue)
    results = [await verify_sentence(client, model, it, turns, chart or {}, source, method) for it in items]
    return apply(note, results), results
