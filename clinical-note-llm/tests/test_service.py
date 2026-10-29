"""API 테스트. 실제 OpenAI 클라이언트 코드 경로를 그대로 타되, 네트워크 대신 가짜 모델 서버(httpx MockTransport)가 답한다."""
import json
import sys
from pathlib import Path

import httpx2 as httpx  # openai 3.x·starlette가 쓰는 httpx 후속 패키지
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service import app as service  # noqa: E402

DIALOGUE = "[doctor] hi, what brings you in today? [patient] i have had knee pain for two weeks."
NOTE = {
    "subjective": "CHIEF COMPLAINT\n\nKnee pain.",
    "objective_exam": "PHYSICAL EXAMINATION\n\nTenderness of the right knee.",
    "objective_results": "",
    "assessment_and_plan": "ASSESSMENT AND PLAN\n\nKnee strain. Ice and rest.",
}


def completion(content, finish_reason="stop"):
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "note-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def sse(content):
    chunks = [content[i:i + 7] for i in range(0, len(content), 7)]
    lines = []
    for c in chunks:
        chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "note-model",
                 "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class FakeLLM:
    """요청을 기록하고, 정해 둔 응답을 순서대로 돌려준다."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def handler(self, request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "note-model", "object": "model",
                                                                          "created": 0, "owned_by": "test"}]})
        body = json.loads(request.content)
        self.requests.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        if body.get("stream"):
            return httpx.Response(200, content=sse(reply), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=completion(*reply) if isinstance(reply, tuple) else completion(reply))


@pytest.fixture
def make_client():
    clients = []

    def make(replies):
        fake = FakeLLM(replies)
        service.app.state.llm = AsyncOpenAI(
            base_url="http://fake/v1", api_key="x", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
        )
        client = TestClient(service.app)
        client.__enter__()
        clients.append(client)
        return client, fake

    yield make
    for c in clients:
        c.__exit__(None, None, None)
    del service.app.state.llm


def test_note_ok(make_client):
    client, fake = make_client([json.dumps(NOTE)])
    r = client.post("/v1/notes", json={"dialogue": DIALOGUE})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["note"] == NOTE
    assert body["attempts"] == 1
    assert body["note_text"].startswith("CHIEF COMPLAINT")
    assert "objective_results" not in body["note_text"]
    # 모델 서버로 간 요청: 학습 때와 같은 프롬프트, 스키마 강제, 검증셋에서 고른 반복 벌점
    sent = fake.requests[0]
    assert sent["messages"][1]["content"] == DIALOGUE
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["frequency_penalty"] == 0.4 and sent["temperature"] == 0


def test_review_flags(make_client):
    # 대화에 없는 나이·검사를 쓰면 확인 표시, 대화에 있는 수치("two weeks" ↔ 2)는 표시하지 않는다
    note = dict(NOTE, subjective="CHIEF COMPLAINT\n\nKnee pain for 2 weeks. He is a pleasant 58-year-old male.",
                objective_results="RESULTS\n\nMRI of the knee shows a meniscal tear. Hemoglobin A1c 7.2.")
    client, _ = make_client([json.dumps(note), json.dumps(NOTE)])
    review = client.post("/v1/notes", json={"dialogue": DIALOGUE}).json()["review"]
    assert review["needs_review"]
    assert review["invented_age"] == ["58"]
    assert review["unsupported_tests"] == ["mri"]
    assert review["unsupported_numbers"] == ["7.2"]
    clean = client.post("/v1/notes", json={"dialogue": DIALOGUE}).json()["review"]
    assert not clean["needs_review"]


def test_chart_input_and_review(make_client):
    # 차트를 주면 학습 때와 같은 "차트 + 대화" 입력, 근거 검사도 차트까지 본다
    note = dict(NOTE, subjective="CHIEF COMPLAINT\n\nKnee pain. Martha is a 58-year-old female.",
                objective_results="RESULTS\n\nHemoglobin A1c 7.2.")
    client, fake = make_client([json.dumps(note), json.dumps(note)])
    chart = {"name": "Martha Lopez", "age": "58", "sex": "Female", "prior_results": ["Hemoglobin A1c 7.2"]}
    review = client.post("/v1/notes", json={"dialogue": DIALOGUE, "chart": chart}).json()["review"]
    sent = fake.requests[0]["messages"]
    assert sent[1]["content"].startswith("PATIENT CHART\nName: Martha Lopez\nAge: 58\n")
    assert sent[1]["content"].endswith("CONVERSATION\n" + DIALOGUE)
    assert not review["needs_review"]  # 나이·수치 모두 차트에 있다
    # 나이 칸이 비어 있는데 나이를 쓰면 표시한다
    review = client.post("/v1/notes", json={"dialogue": DIALOGUE, "chart": dict(chart, age="")}).json()["review"]
    assert review["invented_age"] == ["58"]


def test_truncated_json_retries_once(make_client):
    client, fake = make_client([('{"subjective": "CHIEF COMPLAINT\\n\\nKnee pain. Knee pain. Knee', "length"),
                                json.dumps(NOTE)])
    r = client.post("/v1/notes", json={"dialogue": DIALOGUE})
    assert r.status_code == 200
    assert r.json()["attempts"] == 2
    assert r.json()["completion_tokens"] == 100  # 두 번 생성한 토큰 합
    assert fake.requests[1]["frequency_penalty"] > fake.requests[0]["frequency_penalty"]
    assert client.get("/metrics").json()["retries"] == 1


def test_invalid_twice_is_502(make_client):
    client, _ = make_client(["not json", '{"subjective": "x"}'])  # 두 번째는 JSON이지만 필드가 빠짐
    r = client.post("/v1/notes", json={"dialogue": DIALOGUE})
    assert r.status_code == 502
    assert client.get("/metrics").json()["failures"] == 1


def test_backend_error_is_502(make_client):
    client, _ = make_client([httpx.Response(500, json={"error": "boom"})])
    assert client.post("/v1/notes", json={"dialogue": DIALOGUE}).status_code == 502


def test_input_validation(make_client):
    client, fake = make_client([])
    assert client.post("/v1/notes", json={"dialogue": "hi"}).status_code == 422
    too_long = "[doctor] " + "a" * service.MAX_DIALOGUE_CHARS
    assert client.post("/v1/notes", json={"dialogue": too_long}).status_code == 413
    assert fake.requests == []  # 모델 서버까지 가지 않는다


def test_stream(make_client):
    client, _ = make_client([json.dumps(NOTE)])
    with client.stream("POST", "/v1/notes/stream", json={"dialogue": DIALOGUE}) as r:
        text = "".join(r.iter_text())
    events = [block.split("\n") for block in text.strip().split("\n\n")]
    names = [e[0].removeprefix("event: ") for e in events]
    assert names[-1] == "done" and set(names[:-1]) == {"token"}
    streamed = "".join(json.loads(e[1].removeprefix("data: "))["text"] for e in events[:-1])
    assert json.loads(streamed) == NOTE
    assert json.loads(events[-1][1].removeprefix("data: "))["note"] == NOTE


def test_stream_invalid_json(make_client):
    client, _ = make_client(['{"subjective": "cut off'])
    with client.stream("POST", "/v1/notes/stream", json={"dialogue": DIALOGUE}) as r:
        text = "".join(r.iter_text())
    assert "event: error" in text and "invalid_json" in text


def test_health(make_client):
    client, _ = make_client([])
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["models"] == ["note-model"]
