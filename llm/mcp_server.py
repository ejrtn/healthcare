"""
자가 건강 체크 RAG 챗봇을 MCP(Model Context Protocol) 서버로 노출한다.

main.py가 FastAPI `/chat` 하나만 제공하는 것과 달리, 여기서는 MCP 클라이언트
(Claude Desktop 등)가 이 프로젝트의 KDCA 기반 검색·답변 기능을 표준 프로토콜로
직접 호출할 수 있게 도구(tool) 2개를 노출한다. main.py의 로직을 다시 짜지 않고
이미 로드된 모델/데이터(embed_model, chromadb, llm 등)를 그대로 재사용한다.

**전송 방식: stdio가 아니라 streamable-http를 쓴다.** 처음엔 stdio(Claude
Desktop이 이 스크립트를 그때그때 새 프로세스로 실행)로 만들었는데, 실측해보니
Claude Desktop이 도구 호출에 대한 응답을 기다리는 시간에 제한이 있어서, 8B
모델을 로딩하는 90초 이상을 못 버티고 "Request timed out"으로 계속 실패했다.
별도 스레드로 모델을 미리 로딩해보는 것도 시도했지만, llama-cpp-python의 모델
로딩이 GIL을 점유하는 것으로 보여 스레드로는 실질적인 개선이 없었다. 그래서
이 프로세스를 API 서버처럼 미리 띄워두고 계속 켜두는 방식(streamable-http)으로
바꿨다 — 모델 로딩은 이 서버를 시작할 때 한 번만 겪고, Claude Desktop은 그냥
이미 켜져 있는 서버에 접속만 한다.

두 도구는 성격이 다른 MCP 패턴을 보여준다:
- `search_medical_knowledge`: 검색만 하고 로컬 LLM은 호출하지 않는다. 근거
  청크(질병/카테고리/본문)를 그대로 반환해서, 호출한 쪽(Claude 등)이 직접
  그 근거를 읽고 판단하게 한다 — 이중으로 LLM을 거치지 않는, MCP에서 흔히
  권장되는 "데이터 제공형" 도구 패턴.
- `ask_local_health_bot`: main.chat()을 그대로 호출해서, 검색+로컬 Llama 답변
  생성까지 끝난 최종 응답을 돌려준다 — 기존 서비스를 도구로 감싸는
  "작업 자동화형" 패턴.

설치 및 실행:
    pip install -r requirements.txt "mcp[cli]"
    python mcp_server.py               # http://127.0.0.1:8001/mcp 로 계속 켜둔 채 실행
                                        # (main.py의 웹 UI가 쓰는 8000번과 겹치지 않음 —
                                        # 둘을 동시에 띄워도 됨)

Claude Desktop 설정 예시(claude_desktop_config.json) — 이제 프로세스를 실행하는
방식(command/args)이 아니라, 이미 켜져 있는 서버에 URL로 접속하는 방식이다:
    {
      "mcpServers": {
        "self-health-check-rag": {
          "url": "http://127.0.0.1:8001/mcp"
        }
      }
    }

주의: 이 설정이 동작하려면 `python mcp_server.py`가 먼저 실행되어 있어야 한다
— Claude Desktop이 알아서 띄워주지 않는다(stdio 방식과의 가장 큰 차이).

테스트: Claude Desktop 없이 바로 확인하려면 서버를 먼저 띄운 뒤
`test_mcp_http_client.py`(HTTP용, 있다면)를 쓰거나, 브라우저·curl로
`http://127.0.0.1:8001/mcp`에 MCP 프로토콜 요청을 보내 확인할 수 있다.
stdio 시절 만든 `test_mcp_client.py`는 이제 이 서버와 맞지 않으니 참고만 할 것.
"""
import threading
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

# main.py를 모듈 로드 시점에 바로 import하면, 8B 모델·임베딩·리랭커가 전부
# 로딩된 뒤에야 이 프로세스가 MCP 초기화 요청에 응답할 수 있게 된다. 실측 확인
# (1차): Claude Desktop은 서버가 연결에 응답하기까지 기다리는 시간에 제한이
# 있는데, 그 제한을 넘겨서 "Request timed out"으로 연결 자체가 실패했다.
#
# 1차 수정(첫 도구 호출 시점에 지연 로딩)으로 연결은 바로 됐지만, 실측(2차)에서
# 다른 타임아웃에 또 걸렸다 — 이번엔 "도구 호출 자체"에 대한 응답 제한 시간을
# 넘긴 것("Search medical knowledge 사용함 → Error: Request timed out"). 즉
# "언제 로딩을 시작하느냐"가 아니라 "로딩을 끝낸 상태에서 호출을 받아야 한다"는
# 게 핵심이다. 그래서 프로세스가 시작되자마자(=stdio 핸드셰이크와 동시에)
# 백그라운드 스레드에서 미리 로딩을 시작해둔다 — 사용자가 Claude Desktop에서
# 질문을 입력하는 몇 초 동안 로딩이 끝나 있을 가능성이 높고, 그래도 다 못
# 끝났으면 get_main() 호출부에서 로딩이 끝날 때까지만 기다린다(중복 로딩 방지).
_main = None
_main_lock = threading.Lock()


def get_main():
    global _main
    with _main_lock:
        if _main is None:
            import main as _main_module
            _main = _main_module
    return _main


def _warm_up():
    """
    main.py도 이제 embed_model/reranker(빠름, ~2초)와 llm(8B, 100초 이상)을
    분리해서 지연 로딩한다(main.py의 get_llm() 참고) — search_medical_knowledge
    처럼 LLM이 필요 없는 도구가 그 로딩을 기다리지 않게 하기 위해서다. 그래서
    여기서도 main을 불러온 뒤, 검색용 모델은 이미 main import 시점에 로딩되고,
    LLM은 별도로 한 번 더 예열해준다 — ask_local_health_bot을 처음 쓸 때도
    가능한 한 이미 로딩이 끝나 있도록.
    """
    m = get_main()
    m.get_llm()


threading.Thread(target=_warm_up, daemon=True).start()


server = MCPServer(
    name="self-health-check-rag",
    instructions=(
        "질병관리청(KDCA) 공개 건강정보 1,145건 기반 자가 건강 체크 RAG 챗봇. "
        "정확한 진단이 아니라 1차 참고용 정보 제공이 목적이며, 실제 진단은 "
        "반드시 의료진과 상담해야 한다."
    ),
)


def _pack_results(docs: List[str], distances: List[float], metadatas: List[dict]) -> List[Dict[str, Any]]:
    return [
        {"disease": meta.get("disease", ""), "category": meta.get("category", ""), "content": doc, "distance": dist}
        for doc, dist, meta in zip(docs, distances, metadatas)
    ]


@server.tool()
async def search_medical_knowledge(question: str) -> Dict[str, Any]:
    """
    질문에서 병명을 감지하면 그 병명 범위로, 감지하지 못하면(증상 기반 질문)
    전체 검색으로 KDCA 건강정보를 찾아 근거 청크를 그대로 반환한다. 로컬 LLM은
    호출하지 않는다 — 답변 생성은 호출한 쪽(예: Claude)이 이 근거를 보고 직접
    한다. main.py의 THRESHOLD(코사인 거리 0.5)를 넘는 애매한 결과는 "관련
    없음"으로 정직하게 반환한다.
    """
    main = get_main()
    query_embedding = main.embed_model.encode([question]).tolist()
    diseases = main.detect_diseases(question)
    category_keywords = main.detect_category_keywords(question)

    if diseases:
        results: List[Dict[str, Any]] = []
        for disease in diseases:
            docs, distances, metadatas = main.search_disease(query_embedding, disease, category_keywords, 3)
            if not docs or distances[0] > main.THRESHOLD:
                continue
            docs, distances, metadatas = main.rerank(question, docs, distances, metadatas, 3)
            results.extend(_pack_results(docs, distances, metadatas))
        if not results:
            return {"found": False, "message": "관련 정보를 찾지 못했습니다.", "results": []}
        return {"found": True, "results": results}

    # 병명이 감지되지 않은 경우(증상 기반 질문): main.py의 /chat과 달리 이 도구는
    # history가 없는 단발성 호출이라, 멀티턴 누적·투표 로직(vote_candidates_by_symptom)
    # 대신 단순 벡터 검색으로 충분하다 — main.py 리팩터링으로 SYMPTOM_CANDIDATES
    # 상수가 사라져서 직접 숫자(15)를 쓴다.
    docs, distances, metadatas = main.vector_search(query_embedding, n_results=15)
    if not docs or distances[0] > main.THRESHOLD:
        return {"found": False, "message": "관련 정보를 찾지 못했습니다.", "results": []}
    docs, distances, metadatas = main.rerank(question, docs, distances, metadatas, 3)
    return {"found": True, "results": _pack_results(docs, distances, metadatas)}


@server.tool()
async def ask_local_health_bot(message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    main.py의 /chat과 완전히 동일한 파이프라인(검색 → 리랭킹 → 로컬 Llama-3.1-8B
    답변 생성)을 그대로 호출해 최종 답변을 반환한다. history는 [{"role":
    "user"|"assistant", "content": str}, ...] 형식으로, 멀티턴 증상 문진을
    이어가려면 이전 대화를 그대로 다시 넘겨야 한다(서버가 상태를 갖지 않기
    때문 — main.py와 동일).
    """
    main = get_main()
    request = main.ChatRequest(message=message, history=history or [])
    return await main.chat(request)


if __name__ == "__main__":
    # stdio 대신 streamable-http로 띄운다 — stdio는 Claude Desktop이 도구를 쓸
    # 때마다 이 프로세스를 새로 실행해서 8B 모델을 매번 다시 로딩하게 만든다
    # (실측: 첫 호출까지 90초 이상). streamable-http는 이 프로세스를 API
    # 서버처럼 한 번 띄워두고 계속 켜두는 방식이라, 모델 로딩은 이 서버를 시작할
    # 때 딱 한 번만 겪는다 — 이후 Claude Desktop을 몇 번을 껐다 켜도 이 서버
    # 프로세스 자체가 살아있는 한 다시 로딩하지 않는다.
    #
    # 포트는 main.py(FastAPI 웹 UI, 8000)와 겹치지 않게 8001을 쓴다 — 웹 챗봇과
    # 이 MCP 서버를 동시에 띄워도 충돌하지 않는다.
    server.run(transport="streamable-http", host="127.0.0.1", port=8001)
