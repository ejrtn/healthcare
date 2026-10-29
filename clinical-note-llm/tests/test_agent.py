"""근거 검색·에이전트 테스트. 모델 서버는 가짜(httpx MockTransport)로, OpenAI 클라이언트 코드 경로는 그대로 탄다."""
import asyncio
import json
import sys
from pathlib import Path

import httpx2 as httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import apply_divisions, flagged_sentences, revise  # noqa: E402
from evidence import search, split_turns  # noqa: E402
from faithfulness import check  # noqa: E402

DIALOGUE = ("[doctor] how is your blood pressure at home ?\n[patient] it's been high , like one fifty .\n"
            "[doctor] okay , i'm gon na go ahead and start you on some hydrochlorothiazide , it's a pretty low dose .\n"
            "[doctor] and i'll increase your metformin to one thousand milligrams twice a day .")
NOTE = ("Blood pressure has been elevated at home. Start hydrochlorothiazide 12.5 mg daily. "
        "Increase metformin to 1000 mg twice daily.")


def test_search_finds_turn_by_term_and_spoken_number():
    turns = split_turns(DIALOGUE)
    assert search("hydrochlorothiazide", turns, method="bm25")[0]["turn"] == 2
    assert search("metformin 1000 mg", turns, method="bm25")[0]["turn"] == 3  # "one thousand"도 1000으로 색인


def test_flagged_sentences_pick_only_unsupported_values():
    items = flagged_sentences(NOTE, check(NOTE, DIALOGUE))
    assert items == [{"sentence": "Start hydrochlorothiazide 12.5 mg daily.", "values": ["12.5"]}]


def tool_call(name, args):
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def answer(content):
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}


def run(replies):
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        reply = replies.pop(0)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "agent", **reply})

    client = AsyncOpenAI(base_url="http://fake/v1", api_key="x", max_retries=0,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    note, results = asyncio.run(revise(client, "agent", NOTE, DIALOGUE, chart={"medications": ["metformin 500 mg"]}))
    return note, results, sent


def test_agent_searches_then_fixes_unsupported_dose():
    fixed = "Start hydrochlorothiazide at a low dose (dose not stated)."
    note, results, sent = run([
        tool_call("search_dialogue", {"query": "hydrochlorothiazide"}),
        tool_call("get_chart", {"field": "medications"}),
        answer(""),
        answer(json.dumps({"verdict": "fix", "new_text": fixed, "evidence": [2]})),
    ])
    assert fixed in note and "12.5" not in note and "1000 mg" in note  # 근거 있는 문장은 건드리지 않는다
    r = results[0]
    assert r["applied"] and [c["tool"] for c in r["tool_calls"]] == ["search_dialogue", "get_chart"]
    assert r["tool_calls"][0]["result"].startswith("#2 [doctor] okay , i'm gon na go ahead and start you on some hydro")
    # 도구 결과가 다음 요청에 tool 메시지로 돌아가고, 마지막 판단은 JSON 스키마로 강제된다
    assert sent[1]["messages"][-1]["role"] == "tool" and sent[1]["tools"]
    assert sent[-1]["response_format"]["type"] == "json_schema" and "tools" not in sent[-1]


def test_fix_that_still_invents_a_value_is_not_applied():
    note, results, _ = run([
        answer(""),
        answer(json.dumps({"verdict": "fix", "new_text": "Start hydrochlorothiazide 25 mg daily.", "evidence": []})),
    ])
    assert note == NOTE and results[0]["reason"] == "fix_still_unsupported"


def test_apply_divisions_edits_only_the_sentence_section():
    divisions = {"subjective": "Knee pain.", "assessment_and_plan": "Start hydrochlorothiazide 12.5 mg daily. Recheck."}
    r = {"sentence": "Start hydrochlorothiazide 12.5 mg daily.", "verdict": "remove", "new_text": "", "applied": True}
    assert apply_divisions(divisions, [r]) == {"subjective": "Knee pain.", "assessment_and_plan": "Recheck."}
