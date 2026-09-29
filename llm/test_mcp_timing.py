"""
Claude Desktop과 똑같은 타이밍(연결 직후 몇 초 뒤 질문)을 재현해서, 백그라운드
프리로딩이 실제로 시간을 얼마나 절약하는지, 그리고 도구 호출 자체가 얼마나
걸리는지 재는 진단 스크립트.
"""
import asyncio
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    server_params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    t0 = time.time()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"[{time.time()-t0:.1f}s] 초기화 완료 (연결)")

            tools = await session.list_tools()
            print(f"[{time.time()-t0:.1f}s] tools/list 완료: {[t.name for t in tools.tools]}")

            # Claude Desktop처럼 사용자가 질문을 타이핑하는 시간을 흉내낸다
            print("5초 대기 (질문 타이핑 흉내)...")
            await asyncio.sleep(5)

            t1 = time.time()
            print(f"[{time.time()-t0:.1f}s] search_medical_knowledge 호출 시작")
            result = await session.call_tool(
                "search_medical_knowledge", {"question": "고혈압 원인이 뭐야?"}
            )
            print(f"[{time.time()-t0:.1f}s] 응답 완료 (호출 자체 소요: {time.time()-t1:.1f}s)")
            for block in result.content:
                text = getattr(block, "text", str(block))
                print(text[:200])


if __name__ == "__main__":
    asyncio.run(main())
