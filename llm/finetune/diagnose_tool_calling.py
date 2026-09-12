"""
agentic_experiment.py에서 tool calling이 0% 정확도로 완전히 실패한 원인을 진단한다.

가설을 하나씩 분리해서 확인:
1. 필드 1개짜리 tool 1개 — 모델이 구조화 출력 자체를 못 하는지 확인
2. 필드 2개짜리 tool 1개 (병명+카테고리 동시) — agentic_experiment.py와 같은 조건 재현
3. 필드 1개짜리 tool을 2번 순차 호출 — 필드를 쪼개면 회복되는지 확인

실행:
    cd project
    python finetune/diagnose_tool_calling.py
"""
import sys
from llama_cpp import Llama

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH = "models/meta-llama-3.1-8b-instruct-q4_k_m.gguf"


def load_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=14,
        n_threads_batch=14,
        chat_format="chatml-function-calling",
        verbose=False,
    )


def test_single_field(llm):
    print("=== 1. 필드 1개짜리 tool 1개 ===")
    tool = [{
        "type": "function",
        "function": {
            "name": "echo_disease_name",
            "description": "질문에 나온 병명을 그대로 돌려준다",
            "parameters": {
                "type": "object",
                "properties": {"disease_name": {"type": "string"}},
                "required": ["disease_name"],
            },
        },
    }]
    for q in ["간흡충증이 뭐야?", "고혈압 원인이 뭐야?", "서울 날씨 알려줘"]:
        r = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "질문에 나온 병명을 echo_disease_name 함수로 그대로 돌려주세요."},
                {"role": "user", "content": q},
            ],
            tools=tool, tool_choice="auto",
        )
        print(f"[{q}] ->", r["choices"][0]["message"].get("tool_calls"))


def test_two_fields(llm):
    print("\n=== 2. 필드 2개짜리 tool 1개 (병명+카테고리 동시) ===")
    tool = [{
        "type": "function",
        "function": {
            "name": "search_by_disease_category",
            "description": "병의 특정 카테고리 정보를 검색한다",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease": {"type": "string", "description": "정확한 병명"},
                    "category": {"type": "string", "description": "원인, 증상, 치료, 예방, 합병증, 진단, 관리 중 하나"},
                },
                "required": ["disease", "category"],
            },
        },
    }]
    for q in ["당뇨병 무슨 약 먹어?", "고혈압 원인이 뭐야?", "감기 합병증 있어?"]:
        r = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "질문에 맞는 도구를 호출하세요."},
                {"role": "user", "content": q},
            ],
            tools=tool, tool_choice="auto",
        )
        print(f"[{q}] ->", r["choices"][0]["message"].get("tool_calls"))


def test_sequential_single_field(llm):
    print("\n=== 3. 필드 1개짜리 tool을 2번 순차 호출 ===")
    extract_disease = [{"type": "function", "function": {
        "name": "extract_disease", "description": "질문에서 병명만 추출한다",
        "parameters": {"type": "object", "properties": {"disease": {"type": "string"}}, "required": ["disease"]},
    }}]
    extract_category = [{"type": "function", "function": {
        "name": "extract_category", "description": "질문의 의도가 원인/증상/치료/예방/합병증/진단/관리 중 무엇인지 고른다",
        "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]},
    }}]
    for q in ["당뇨병 무슨 약 먹어?", "고혈압 원인이 뭐야?", "감기 합병증 있어?"]:
        r1 = llm.create_chat_completion(
            messages=[{"role": "system", "content": "질문에서 병명을 추출하세요."}, {"role": "user", "content": q}],
            tools=extract_disease, tool_choice="auto",
        )
        r2 = llm.create_chat_completion(
            messages=[{"role": "system", "content": "질문의 의도(원인/증상/치료/예방/합병증/진단/관리)를 고르세요."}, {"role": "user", "content": q}],
            tools=extract_category, tool_choice="auto",
        )
        print(f"[{q}]")
        print("  disease:", r1["choices"][0]["message"].get("tool_calls"))
        print("  category:", r2["choices"][0]["message"].get("tool_calls"))


if __name__ == "__main__":
    llm = load_llm()
    test_single_field(llm)
    test_two_fields(llm)
    test_sequential_single_field(llm)
