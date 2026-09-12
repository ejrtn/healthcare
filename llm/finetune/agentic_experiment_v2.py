"""
agentic_experiment.py에서 발견한 문제(필드 2개 동시 추출 실패, 어쩌면 tool 2개
동시 제공도 원인)를 반영해서 라우팅을 재설계했다:

  1단계: classify_question_type 딱 1개 tool, 필드 1개(type) — 병명 언급 질문인지
         증상만 있는 질문인지 LLM이 스스로 분류
  2단계: 분류 결과에 따라
         - 병명형: extract_disease(1개 tool, 필드 1개) -> extract_category(1개 tool, 필드 1개) 순차 호출
         - 증상형: extract_symptom_text(1개 tool, 필드 1개) 호출

모든 호출에서 tool은 항상 1개만 주어지고, 필드도 항상 1개뿐이다. 이렇게 해도
규칙 기반을 못 따라가는지, 아니면 0%에서 회복되는지 15~16문항 전체로 확인한다.

실행:
    cd project
    python finetune/agentic_experiment_v2.py
"""
import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "eval"))
sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from llama_cpp import Llama  # noqa: E402
import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from ground_truth import GROUND_TRUTH, SYMPTOM_CASES  # noqa: E402

DISEASE_INDICES = [0, 2, 3, 5, 7, 11, 16, 23]

SYMPTOM_TESTS = list(SYMPTOM_CASES) + [
    ("허리가 아파요", ["요통", "강직성척추염", "추간판탈출증(디스크)", "강직척추염", "염좌", "척추관 협착증", "척추의 형태 이상(척추후만증)"]),
    ("머리가 어지럽다", ["어지럼", "실신", "빈혈"]),
]

CLASSIFY_TOOL = [{"type": "function", "function": {
    "name": "classify_question_type",
    "description": "질문이 병명을 언급하는지, 증상만 설명하는지 분류한다",
    "parameters": {"type": "object", "properties": {
        "type": {"type": "string", "enum": ["disease_named", "symptom_only"]},
    }, "required": ["type"]},
}}]
EXTRACT_DISEASE_TOOL = [{"type": "function", "function": {
    "name": "extract_disease", "description": "질문에서 병명만 추출한다",
    "parameters": {"type": "object", "properties": {"disease": {"type": "string"}}, "required": ["disease"]},
}}]
EXTRACT_CATEGORY_TOOL = [{"type": "function", "function": {
    "name": "extract_category", "description": "질문의 의도가 원인/증상/치료/예방/합병증/진단/관리 중 무엇인지 고른다",
    "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]},
}}]
EXTRACT_SYMPTOM_TOOL = [{"type": "function", "function": {
    "name": "extract_symptom_text", "description": "핵심 증상을 정리한 검색용 문구를 만든다",
    "parameters": {"type": "object", "properties": {"symptom_text": {"type": "string"}}, "required": ["symptom_text"]},
}}]


def call_tool(llm, system_prompt, question, tool):
    t0 = time.time()
    try:
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            tools=tool, tool_choice="auto",
        )
        elapsed = time.time() - t0
        calls = r["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            return None, elapsed, "tool 호출 안 함"
        args = json.loads(calls[0]["function"]["arguments"])
        return args, elapsed, None
    except Exception as e:
        return None, time.time() - t0, f"오류: {e}"


def agentic_route_v2(llm, question):
    """1단계 분류 -> 2단계(병명형: 순차 2회 / 증상형: 1회)."""
    total_time = 0.0
    args, t, err = call_tool(llm, "질문이 병명을 언급하는지(disease_named), 증상만 설명하는지(symptom_only) 분류하세요.", question, CLASSIFY_TOOL)
    total_time += t
    if err or not args:
        return {"kind": "unknown"}, total_time, err or "분류 실패"

    qtype = args.get("type")
    if qtype == "disease_named":
        d_args, t1, err1 = call_tool(llm, "질문에서 병명을 추출하세요.", question, EXTRACT_DISEASE_TOOL)
        c_args, t2, err2 = call_tool(llm, "질문의 의도(원인/증상/치료/예방/합병증/진단/관리)를 고르세요.", question, EXTRACT_CATEGORY_TOOL)
        total_time += t1 + t2
        return {"kind": "disease_named", "disease": d_args.get("disease") if d_args else None,
                "category": c_args.get("category") if c_args else None}, total_time, err1 or err2
    else:
        s_args, t1, err1 = call_tool(llm, "핵심 증상을 정리한 검색용 문구를 만드세요.", question, EXTRACT_SYMPTOM_TOOL)
        total_time += t1
        return {"kind": "symptom_only", "symptom_text": s_args.get("symptom_text") if s_args else None}, total_time, err1


def category_matches(predicted, expected_list):
    if not predicted:
        return False
    return any(predicted in e or e in predicted for e in expected_list)


def main_run():
    print("에이전틱 v2 라우팅용 LLM 로딩 (chatml-function-calling)...")
    llm = Llama(
        model_path="models/meta-llama-3.1-8b-instruct-q4_k_m.gguf",
        n_ctx=4096, n_threads=14, n_threads_batch=14,
        chat_format="chatml-function-calling", verbose=False,
    )
    print("임베딩 모델/ChromaDB 로딩...")
    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    chroma_client = chromadb.PersistentClient(path=os.path.join(PROJECT_ROOT, "vector_db"))
    collection = chroma_client.get_collection("medical_knowledge_v2")

    def vector_search_top_disease(text):
        emb = embed_model.encode([text]).tolist()
        res = collection.query(query_embeddings=emb, n_results=1, include=["metadatas"])
        return res["metadatas"][0][0].get("disease") if res["metadatas"][0] else None

    agentic_correct = 0
    agentic_time_total = 0.0
    rows = []

    for idx in DISEASE_INDICES:
        question, expected_disease, expected_categories = GROUND_TRUTH[idx]
        result, elapsed, err = agentic_route_v2(llm, question)
        ok = (result.get("kind") == "disease_named" and result.get("disease") == expected_disease
              and category_matches(result.get("category"), expected_categories))
        agentic_correct += ok
        agentic_time_total += elapsed
        rows.append({"type": "disease", "question": question, "expected_disease": expected_disease,
                     "expected_categories": expected_categories, "result": result, "time": elapsed, "ok": ok, "error": err})
        print(f"[병명형] {question} -> {result} ({elapsed:.1f}s) {'OK' if ok else 'FAIL'}{' ('+err+')' if err else ''}")

    for question, acceptable in SYMPTOM_TESTS:
        result, elapsed, err = agentic_route_v2(llm, question)
        if result.get("kind") == "symptom_only" and result.get("symptom_text"):
            top = vector_search_top_disease(result["symptom_text"])
        elif result.get("kind") == "disease_named" and result.get("disease"):
            top = result["disease"]
        else:
            top = None
        ok = top in acceptable
        agentic_correct += ok
        agentic_time_total += elapsed
        rows.append({"type": "symptom", "question": question, "acceptable": acceptable,
                     "result": result, "top_disease": top, "time": elapsed, "ok": ok, "error": err})
        print(f"[증상형] {question} -> {result} -> top={top} ({elapsed:.1f}s) {'OK' if ok else 'FAIL'}{' ('+err+')' if err else ''}")

    total = len(DISEASE_INDICES) + len(SYMPTOM_TESTS)
    print()
    print("=" * 80)
    print(f"에이전틱 v2(순차 단일필드) 정확도: {agentic_correct}/{total} ({agentic_correct/total:.1%}), 총 소요시간: {agentic_time_total:.1f}s")
    print("(참고: 규칙 기반은 이전 실험에서 12~13/15~16 = 75~87%, 3ms~5.6s)")
    print("(참고: 에이전틱 v1(필드 2개 동시)은 0/15~16 = 0%, 167~917s)")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_results_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"agentic_correct": agentic_correct, "total": total,
                    "agentic_time_total": agentic_time_total, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 {out_path}에 저장되었습니다.")


if __name__ == "__main__":
    main_run()
