"""진료기록 생성 API (FastAPI, 비동기).

모델 서버와는 OpenAI 호환 API로만 통신한다. 그래서 같은 코드로
  - GPU: vLLM (`vllm serve`)
  - CPU: llama.cpp (`llama-server`, GGUF)
를 환경변수(LLM_BASE_URL)만 바꿔 갈아 끼운다.

엔드포인트
  POST /v1/notes          대화 → 4구역 JSON 진료기록 (스키마 검증, 실패 시 1회 재시도)
  POST /v1/notes/stream   같은 생성을 토큰 단위로 스트리밍 (SSE)
  GET  /health            모델 서버 연결 확인
  GET  /metrics           요청 수·실패 수·지연 시간

실행:
    LLM_BASE_URL=http://127.0.0.1:8000/v1 uvicorn service.app:app --port 8080
"""
import asyncio
import json
import os
import statistics
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from prompts import build_messages  # noqa: E402
from schema import NOTE_JSON_SCHEMA, render_note  # noqa: E402

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "note-model")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "unused")
# 모델 서버가 감당할 동시 요청 수. 넘치면 API 앞에서 기다리게 해 모델 서버 큐가 무한정 길어지지 않게 한다
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "16"))
QUEUE_TIMEOUT_S = float(os.environ.get("QUEUE_TIMEOUT_S", "30"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "300"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
# 검증셋에서 고른 반복 억제 값 (kaggle/04_decode_sweep.py)
FREQUENCY_PENALTY = float(os.environ.get("FREQUENCY_PENALTY", "0.4"))
# 학습·평가 데이터의 최대 대화가 약 3,000단어(약 4,000토큰). 모델 문맥(8,192) 안에 출력 자리를 남긴다
MAX_DIALOGUE_CHARS = int(os.environ.get("MAX_DIALOGUE_CHARS", "24000"))


class NoteRequest(BaseModel):
    dialogue: str = Field(min_length=20, description="의사-환자 대화 전사본 ([doctor] ... [patient] ... 형식)")


class ClinicalNote(BaseModel):
    subjective: str
    objective_exam: str
    objective_results: str
    assessment_and_plan: str


class NoteResponse(BaseModel):
    note: ClinicalNote
    note_text: str
    model: str
    attempts: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int


class Metrics:
    def __init__(self):
        self.requests = 0
        self.failures = 0
        self.retries = 0
        self.rejected = 0
        self.inflight = 0
        self.latencies = deque(maxlen=1000)

    def snapshot(self):
        lat = sorted(self.latencies)
        pct = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 3)) if lat else (lambda q: None)
        return {
            "requests": self.requests, "failures": self.failures, "retries": self.retries,
            "rejected": self.rejected, "inflight": self.inflight,
            "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
            "latency_p95_s": pct(0.95),
        }


def response_format():
    return {"type": "json_schema", "json_schema": {"name": "clinical_note", "schema": NOTE_JSON_SCHEMA}}


def sampling_kwargs():
    return dict(model=LLM_MODEL, temperature=0, max_tokens=MAX_TOKENS,
                frequency_penalty=FREQUENCY_PENALTY, response_format=response_format())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 테스트에서는 app.state.llm을 미리 넣어 둔다 (가짜 모델 서버)
    if not hasattr(app.state, "llm"):
        app.state.llm = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                                    timeout=REQUEST_TIMEOUT_S, max_retries=0)
    app.state.slots = asyncio.Semaphore(MAX_INFLIGHT)
    app.state.metrics = Metrics()
    yield
    await app.state.llm.close()


app = FastAPI(title="Clinical Note LLM", version="0.1.0", lifespan=lifespan)


@asynccontextmanager
async def acquire_slot(request: Request):
    """동시 처리 상한. 자리가 QUEUE_TIMEOUT_S 안에 안 나면 503으로 바로 거절한다 (클라이언트가 재시도 판단)."""
    state = request.app.state
    try:
        await asyncio.wait_for(state.slots.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        state.metrics.rejected += 1
        raise HTTPException(503, "서버가 바쁩니다. 잠시 후 다시 시도하세요.", headers={"Retry-After": "5"})
    state.metrics.inflight += 1
    try:
        yield
    finally:
        state.metrics.inflight -= 1
        state.slots.release()


def check_length(dialogue: str):
    if len(dialogue) > MAX_DIALOGUE_CHARS:
        raise HTTPException(413, f"대화가 너무 깁니다 ({len(dialogue)}자, 최대 {MAX_DIALOGUE_CHARS}자)")


def parse_note(text: str) -> ClinicalNote:
    return ClinicalNote.model_validate(json.loads(text))


@app.post("/v1/notes", response_model=NoteResponse)
async def create_note(req: NoteRequest, request: Request):
    check_length(req.dialogue)
    state = request.app.state
    state.metrics.requests += 1
    messages = build_messages(req.dialogue, json_mode=True)
    start = time.perf_counter()
    prompt_tokens = completion_tokens = 0

    async with acquire_slot(request):
        # 스키마 강제(structured output)가 있어도 max_tokens에 걸리면 JSON이 닫히지 않는다.
        # 그런 경우 한 번만 다시 시도하고(반복 벌점을 올려서), 그래도 실패하면 502로 알린다.
        for attempt in (1, 2):
            kwargs = sampling_kwargs()
            if attempt == 2:
                kwargs["frequency_penalty"] = min(1.0, FREQUENCY_PENALTY + 0.3)
                state.metrics.retries += 1
            try:
                resp = await state.llm.chat.completions.create(messages=messages, **kwargs)
            except (APIConnectionError, APITimeoutError) as e:
                state.metrics.failures += 1
                raise HTTPException(503, f"모델 서버에 연결할 수 없습니다: {type(e).__name__}")
            except APIStatusError as e:
                state.metrics.failures += 1
                raise HTTPException(502, f"모델 서버 오류 ({e.status_code})")

            if resp.usage:
                prompt_tokens += resp.usage.prompt_tokens
                completion_tokens += resp.usage.completion_tokens
            try:
                note = parse_note(resp.choices[0].message.content or "")
                break
            except (json.JSONDecodeError, ValidationError):
                if attempt == 2:
                    state.metrics.failures += 1
                    raise HTTPException(502, "모델 출력이 진료기록 형식(JSON)이 아닙니다. "
                                             f"finish_reason={resp.choices[0].finish_reason}")

    latency = time.perf_counter() - start
    state.metrics.latencies.append(latency)
    return NoteResponse(
        note=note, note_text=render_note(note.model_dump()), model=resp.model, attempts=attempt,
        latency_s=round(latency, 3), prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


@app.post("/v1/notes/stream")
async def stream_note(req: NoteRequest, request: Request):
    """생성 중인 JSON 조각을 SSE로 바로 흘려보낸다. 마지막에 검증 결과(done/error) 이벤트를 보낸다.

    진료기록 한 건은 수십 초가 걸려서, 화면에 바로 글자가 나타나야 사용자가 기다릴 수 있다.
    """
    check_length(req.dialogue)
    state = request.app.state
    state.metrics.requests += 1
    messages = build_messages(req.dialogue, json_mode=True)

    def event(name, data):
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def events():
        start = time.perf_counter()
        parts = []
        try:
            async with acquire_slot(request):
                stream = await state.llm.chat.completions.create(messages=messages, stream=True, **sampling_kwargs())
                async for chunk in stream:
                    if await request.is_disconnected():  # 사용자가 떠나면 모델 서버 쪽 생성도 끊는다
                        await stream.close()
                        return
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        parts.append(delta)
                        yield event("token", {"text": delta})
        except HTTPException as e:  # 스트림이 이미 시작돼 상태 코드를 바꿀 수 없으므로 이벤트로 알린다
            yield event("error", {"error": "busy", "detail": e.detail})
            return
        except (APIConnectionError, APITimeoutError, APIStatusError) as e:
            state.metrics.failures += 1
            yield event("error", {"error": type(e).__name__})
            return
        try:
            note = parse_note("".join(parts))
        except (json.JSONDecodeError, ValidationError):
            state.metrics.failures += 1
            yield event("error", {"error": "invalid_json"})
            return
        latency = time.perf_counter() - start
        state.metrics.latencies.append(latency)
        yield event("done", {"note": note.model_dump(), "note_text": render_note(note.model_dump()),
                             "latency_s": round(latency, 3)})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
async def health(request: Request):
    try:
        models = await request.app.state.llm.models.list()
    except Exception as e:  # 어떤 이유든 모델 서버를 못 쓰면 준비 안 됨
        raise HTTPException(503, f"모델 서버 연결 실패: {type(e).__name__}")
    return {"status": "ok", "llm_base_url": LLM_BASE_URL, "models": [m.id for m in models.data]}


@app.get("/metrics")
async def metrics(request: Request):
    return request.app.state.metrics.snapshot() | {"max_inflight": MAX_INFLIGHT}
