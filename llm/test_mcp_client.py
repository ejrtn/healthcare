"""
mcp_server.py를 Node.js(MCP Inspector)나 Claude Desktop 없이 직접 테스트하는
스크립트.

`mcp_server.py`를 서브프로세스로 띄워서 stdio로 붙은 뒤, main 모듈을 mock으로
치환했던 이전 검증과 달리 **진짜 MCP 프로토콜**(세션 초기화 핸드셰이크,
tools/list, tools/call 메시지 왕복)까지 그대로 거친다 — `mcp` 패키지가 이미
`requirements.txt`에 있으므로 Node.js/Claude Desktop 설치 없이 바로 실행 가능.

실행:
    cd llm
    python test_mcp_client.py

주의: mcp_server.py가 main.py를 그대로 import하므로, 세션 초기화 시점에 8B
모델과 임베딩 모델이 로드된다 — main.py를 직접 돌릴 때와 마찬가지로 첫 응답까지
수십 초 걸릴 수 있다.
"""
import asyncio

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("세션 초기화 중... (main.py가 모델을 로드해서 수십 초 걸릴 수 있음)")
            await session.initialize()
            print("초기화 완료\n")

            tools = await session.list_tools()
            print("등록된 도구:", [t.name for t in tools.tools])

            print("\n--- search_medical_knowledge 호출 (로컬 LLM 호출 없음, 검색만) ---")
            result = await session.call_tool(
                "search_medical_knowledge", {"question": "고혈압 원인이 뭐야?"}
            )
            for block in result.content:
                print(getattr(block, "text", block))

            print("\n--- ask_local_health_bot 호출 (로컬 8B 답변 생성 — 시간이 걸릴 수 있음) ---")
            result = await session.call_tool(
                "ask_local_health_bot", {"message": "고혈압 원인이 뭐야?"}
            )
            for block in result.content:
                print(getattr(block, "text", block))


if __name__ == "__main__":
    asyncio.run(main())
