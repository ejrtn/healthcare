# 자가 건강 체크 RAG 챗봇

질병관리청(KDCA) 공개 건강정보 1,145건을 벡터DB에 넣고, 병명/카테고리를 정확히
라우팅해서 답하는 로컬 RAG 의료 정보 챗봇. LLM은 100% 로컬(llama-cpp-python,
Llama-3.1-8B-Instruct Q4_K_M)로 돌아가고, 외부 API 호출이 없다.

이 프로젝트의 핵심은 "RAG를 만들었다"가 아니라 **각 설계 선택(청크 크기, 카테고리
라우팅 방식, 리랭킹 적용 여부, 안전 임계값, 에이전틱 패턴 도입 여부)을 가정으로
넘기지 않고 직접 측정해서 결정했다**는 점이다. 그 과정에서 나온 실패 사례(리랭킹이
정확도를 낮춘 경우, tool calling이 완전히 망가진 경우)도 그대로 남겨뒀다.

## 데모

```bash
pip install -r requirements.txt   # 아래 "설치" 참고
python main.py                    # 또는 uvicorn main:app --reload
# http://localhost:8000 접속
```

## 아키텍처

```
질병관리청 XML(1,145개)
  → 정제(base64 이미지·HTML·비질병 안내문 제거)
  → 청킹(문장 단위, 400자, 60자 overlap)
  → 임베딩("병명 카테고리: 내용" 형태로 인코딩) → ChromaDB (29,214청크)

질문
  ├─ 병명 언급됨
  │     → 병명+카테고리 메타데이터 필터 → 벡터검색(8개 후보) → 리랭킹 → top-3
  │
  └─ 병명 미언급 (증상 기반)
        → 대화 이력 누적 → 전체 벡터검색(40개 후보) → 후보 병 개수로 분기
              ├─ 10개 초과   → "질병이 너무 많음, 증상 더 알려달라" (되묻기)
              ├─ 3~10개     → 후보 목록 나열 + 추가 증상 요청 (되묻기)
              └─ 3개 미만    → top-3 리랭킹 → 답변 생성
        (되묻기 최대 3회, 그 이후엔 애매해도 "~등 여러 질환이 예상됩니다" 식으로 답변)

→ 코사인 거리 THRESHOLD(0.5) 미달이면 "관련 없음"으로 조기 거절
→ Llama-3.1-8B(로컬)로 답변 생성 + 출처 표시 + 진료과 추천(추정 어투)
```

## 측정으로 검증한 설계 결정 (`eval/`)

| 결정 | 근거 |
|---|---|
| 청크 크기 400자 | 100~600자 스윕 결과 300~500자가 sweet spot, 400이 그 한가운데 (12/12 정확도) |
| 병명+카테고리 메타데이터 필터링 | 임베딩에 "병명 카테고리: 내용"을 붙여 인코딩해야 병명이 본문에 없는 청크도 찾아짐 |
| 리랭킹(cross-encoder) 적용 | 33문항 기준 RAW 90.9% → 리랭킹 후 93.9%로 순개선 (12문항일 땐 반대로 보였는데, 표본을 늘리자 결론이 뒤집힘 — 그 과정도 기록해둠) |
| THRESHOLD 0.7 → 0.5 | 원래 값(0.7)은 완전히 무관한 질문(날씨/주식 등) 거절 테스트에서 0/5 실패. 0.5로 낮춰 5/5로 수정 (관련 있는 질문 1건이 희생되지만 안전이 우선) |
| 안내문 데이터 필터링 | "겨울철 한파대비 건강수칙" 같은 비질병 공공안내문 5건이 병명 목록에 섞여 증상 문진 로직을 오염시킨 걸 발견하고 파싱 단계에서 제거 |

자세한 수치와 재현 스크립트는 **[`eval/RESULTS.md`](eval/RESULTS.md)** 참고.

검색 라우팅 정확도(위 표)와 별개로, 실제 생성된 답변이 검색된 근거에 충실한지
(faithfulness)·질문에 맞는지(answer_relevancy)를 **RAGAS**로 채점하는
`eval/ragas_eval.py`도 있다 — 기본 judge는 외부 API 대신 이 프로젝트가 이미
로드해둔 로컬 Llama-3.1-8B를 그대로 재사용해 "100% 로컬" 철학을 평가
도구에도 유지했다. 다만 로컬 8B judge가 디제너레이션(생성 퇴화)으로 크래시난
적이 있어서, **API 키가 있으면 Gemini/Claude/ChatGPT까지 추가로 붙여서 같은
답변을 여러 judge가 어떻게 다르게 채점하는지 교차 검증**할 수 있게 확장했다
— 키가 없는 judge는 자동으로 건너뛰고, 프로덕션(`main.py`/`mcp_server.py`)은
여전히 외부 API를 쓰지 않는다. 상세 설계 이유와 현재 상태는
`eval/RESULTS.md` 실험 5 참고.

## 파인튜닝 & 에이전틱 패턴 실험 (`finetune/`)

- **임베딩 모델 파인튜닝**: `ko-sroberta-multitask`를 프로젝트 자체 데이터(카테고리
  라우팅 33건 + 증상 감별진단 6건)로 TripletLoss 파인튜닝. train/test 분리로
  일반화 확인 (test 정확도 55.6%→77.8%), 거절 안전성도 재검증(부작용 없음 확인).
- **에이전틱 tool calling 실험**: 규칙 기반 라우팅을 LLM tool calling으로 대체하면
  어떨지 실측. 처음엔 0% 정확도로 완전 실패 → 원인을 "다중 필드 추출 실패"까지
  진단 → 순차 단일 필드 호출로 재설계해 43.8%까지 회복 → 그래도 규칙 기반
  (75~87%)에는 못 미쳐 도입하지 않기로 결정. "왜 안 되는지"까지 파고든 과정 자체가
  이 실험의 핵심 결과물.

자세한 내용은 **[`finetune/RESULTS.md`](finetune/RESULTS.md)** 참고.

## MCP 서버 (`mcp_server.py`)

`main.py`가 FastAPI `/chat` 엔드포인트 하나만 제공하는 것과 별개로, 같은 검색·
답변 기능을 **MCP(Model Context Protocol)** 표준으로도 노출한다 — Claude
Desktop 같은 MCP 클라이언트가 이 프로젝트의 KDCA 기반 지식을 도구처럼 직접
호출할 수 있게 하기 위함이다. 기존 로직을 다시 짜지 않고 main.py가 이미
로드해둔 모델/DB를 그대로 재사용해서 도구 2개로만 감쌌다:

- `search_medical_knowledge`: 검색만 하고 로컬 LLM은 호출하지 않는다. 근거
  청크를 그대로 반환해서 호출한 쪽이 직접 판단하게 하는 "데이터 제공형" 패턴.
- `ask_local_health_bot`: `main.chat()`을 그대로 호출해 검색+로컬 Llama 답변
  생성까지 끝난 결과를 반환하는 "작업 자동화형" 패턴.

```bash
pip install -r requirements.txt   # mcp 포함
python mcp_server.py              # stdio transport로 실행
```

**테스트 방법 3가지** (`mcp dev`는 Node.js, `mcp install`은 Claude Desktop
앱이 따로 필요해서, 둘 다 없는 환경을 위해 `test_mcp_client.py`를 추가로
만들어뒀다):

| 방법 | 추가 설치 | 확인 범위 |
|---|---|---|
| `python test_mcp_client.py` | 없음 (mcp 패키지만) | 진짜 MCP 프로토콜(초기화, tools/list, tools/call)까지 — Node/Claude Desktop 없이도 가능 |
| `mcp dev mcp_server.py` | Node.js | 웹 UI(Inspector)로 파라미터 입력하며 대화형 확인 |
| `mcp install mcp_server.py` | Claude Desktop 앱 | 실제 Claude 대화 흐름 속에서 도구 호출 확인 |

두 패턴을 의도적으로 나눈 이유는 mcp_server.py 자체의 모듈 docstring에 있다.
**검증 완료**: `main`을 mock으로 치환한 로직 테스트에 이어, 실제 8B 모델과
ChromaDB가 로드된 환경에서 `test_mcp_client.py`로 진짜 MCP 프로토콜(세션
초기화 → `tools/list` → `tools/call`) end-to-end까지 확인했다 —
`search_medical_knowledge`("고혈압 원인이 뭐야?")가 실제 청크(거리 0.326,
0.361)를 정확히 반환했고, `ask_local_health_bot`이 그 근거를 바탕으로 로컬
Llama-3.1-8B가 생성한 답변 + 출처를 정상적으로 돌려주는 것까지 확인했다.

이 프로젝트는 텍스트 기반 RAG 포트폴리오다. 의료 **영상**(X-ray, CT) AI
작업은 별도 포트폴리오 [`../image/`](../image/README.md)로 분리해서 관리한다
(둘의 관계는 [최상위 README](../README.md) 참고).

## 프로젝트 구조

```
main.py                    # FastAPI 서버: RAG 파이프라인 + 멀티턴 증상 문진
mcp_server.py               # 같은 RAG 파이프라인을 MCP 도구로 노출
data/
  parse_xml.py              # XML → 청킹된 JSONL + 병명/카테고리 목록
  kdca_health_info/         # 원본 XML (질병관리청)
  medical_knowledge.jsonl   # 청킹된 학습 데이터
static/index.html           # 채팅 UI (대화 이력, 출처 표시)
vector_db/                  # ChromaDB 영구 저장소
models/                     # Llama-3.1-8B GGUF
eval/                       # 정량 평가 프레임워크 (청크 크기, 라우팅, 거절, 증상 매칭)
finetune/                   # 임베딩 파인튜닝 + 에이전틱 실험
```

## 설치

```bash
pip install fastapi uvicorn chromadb sentence-transformers llama-cpp-python

# Windows에서 llama-cpp-python 빌드가 실패하면 미리 컴파일된 wheel 사용:
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

`models/` 폴더에 `meta-llama-3.1-8b-instruct-q4_k_m.gguf`가 있어야 한다. CPU
전용으로 동작하도록 설정되어 있고(`n_gpu_layers=0`), GPU가 있으면 `main.py`의
`n_gpu_layers`를 20~35 정도로 올리면 훨씬 빨라진다.

## 알려진 한계

- **데이터셋에 없는 병명을 거절 못 함**: THRESHOLD는 완전히 무관한 질문은 잘
  걸러내지만, 의료 도메인 안에 있으나 이 KDCA 데이터셋엔 없는 병명(위염,
  불면증 등)은 거리가 가까워서 못 걸러낸다 — 실측 4/4 전부 실패
  (`eval/RESULTS.md` 실험 3 참고).
- 병명 축약형("당뇨" 등) 미인식 — 의도적으로 보류
- 한 문장에 병명별로 다른 의도가 섞인 복합 질문은 의도가 전체에 공통 적용됨
- 진료과 추천은 데이터에 매핑이 없어 LLM의 일반 지식에 의존 (드물게 부정확할 수 있음)
- CPU 추론이라 답변 생성에 수십 초 소요
- RAGAS 생성 품질 평가(`eval/ragas_eval.py`)는 구현은 끝났지만, 로컬에 이 프로젝트의
  런타임 의존성(torch/chromadb/llama-cpp-python)이 설치된 환경에서 실제 8B 모델로
  돌려 수치를 확보하는 건 아직이다 — 배관(judge/embeddings wrapper, 파이프라인 재현)은
  mock으로 검증했지만 진짜 judge 점수는 아직 없다 (`eval/RESULTS.md` 실험 5 참고).

## 데이터 출처

질병관리청(KDCA) 공공데이터포털 공개 건강정보. 의료 자문이 아니며, 정확한 진단은
반드시 의료진과 상담해야 한다.
