"""
자가 건강 체크 RAG 챗봇을 MCP(Model Context Protocol) 서버로 노출한다.

main.py가 FastAPI `/chat` 하나만 제공하는 것과 달리, 여기서는 MCP 클라이언트
(Claude Desktop 등)가 이 프로젝트의 KDCA 기반 검색·답변 기능을 표준 프로토콜로
직접 호출할 수 있게 도구(tool) 2개를 노출한다. main.py의 로직을 다시 짜지 않고
이미 로드된 모델/데이터(embed_model, chromadb, llm 등)를 그대로 재사용한다 —
프로세스를 새로 띄울 때마다 8B GGUF와 임베딩 모델을 다시 로드하게 되는 구조라,
이 서버도 main.py와 마찬가지로 시작에 시간이 걸린다.

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
    python mcp_server.py               # stdio transport로 실행 (Claude Desktop 등에서 사용)

Claude Desktop 설정 예시(claude_desktop_config.json):
    {
      "mcpServers": {
        "self-health-check-rag": {
          "command": "python",
          "args": ["/absolute/path/to/llm/mcp_server.py"]
        }
      }
    }

테스트: Node.js나 Claude Desktop 없이 바로 확인하려면 `test_mcp_client.py`를
실행 — 실제 8B 모델·ChromaDB가 로드된 환경에서 진짜 MCP 프로토콜(세션 초기화,
tools/list, tools/call)로 두 도구를 호출해 정상 동작을 검증했다.
"""
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

import main  # noqa: E402 — embed_model, llm, chromadb, detect_diseases 등을 그대로 재사용

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

    docs, distances, metadatas = main.vector_search(query_embedding, n_results=main.SYMPTOM_CANDIDATES)
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
    request = main.ChatRequest(message=message, history=history or [])
    return await main.chat(request)


if __name__ == "__main__":
    server.run(transport="stdio")
