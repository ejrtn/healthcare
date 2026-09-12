"""
소규모 에이전틱 라우팅 실험: "LLM이 tool calling으로 라우팅을 직접 판단하면
규칙 기반(detect_diseases + detect_category_keywords + 벡터 검색)보다 나은가?"를
15문항(병명 언급 8개 + 증상만 언급 7개)으로 비교한다.

병명 언급 문항: 규칙 기반은 detect_diseases+detect_category_keywords로, 에이전틱은
search_by_disease_category tool로 (병명, 카테고리)를 판단 — 정답 카테고리와 일치하는지 채점.

증상만 언급 문항(병명 없음): 규칙 기반은 원문 그대로 벡터 검색, 에이전틱은
search_by_symptom tool이 뽑아낸 symptom_text로 벡터 검색 — 이렇게 하면 "LLM이 증상
설명을 정리해주는 전처리가 검색에 도움이 되는지"까지 같이 볼 수 있다. 둘 다 최종
top-1 병명이 그 증상의 감별진단 후보군(acceptable_diseases) 안에 있는지로 채점.

정확도뿐 아니라 라우팅 결정에 걸리는 시간도 같이 측정한다 (규칙 기반은 즉시 실행,
tool calling은 LLM 호출 1회가 추가로 필요).

실행:
    cd project
    python finetune/agentic_experiment.py
"""
import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "eval"))
sys.path.insert(0, PROJECT_ROOT)

from llama_cpp import Llama  # noqa: E402
import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from ground_truth import GROUND_TRUTH, SYMPTOM_CASES  # noqa: E402

# main.py를 통째로 import하면 그 안에서 8B LLM을 또 하나 로드해버려서(메모리 2배),
# 여기선 규칙 기반 라우팅에 필요한 순수 로직 + (임베딩 모델/ChromaDB, LLM은 아님)만 가볍게 복제한다.
with open(os.path.join(PROJECT_ROOT, "data", "diseases.json"), encoding="utf-8") as f:
    KNOWN_DISEASES = json.load(f)
with open(os.path.join(PROJECT_ROOT, "data", "disease_categories.json"), encoding="utf-8") as f:
    DISEASE_CATEGORIES = json.load(f)

CATEGORY_INTENTS = {
    "원인": {"triggers": ["원인", "왜 생기", "왜 걸리", "이유"], "keywords": ["원인", "역학", "병태생리"]},
    "증상": {"triggers": ["증상", "증세"], "keywords": ["증상", "종류"]},
    "진단": {"triggers": ["진단", "검사"], "keywords": ["진단", "검사", "평가"]},
    "약물": {"triggers": ["약", "복용", "처방"], "keywords": ["약물"]},
    "치료": {"triggers": ["치료", "낫는", "고치"], "keywords": ["치료"]},
    "예방": {"triggers": ["예방", "조심", "주의"], "keywords": ["예방", "위험요인"]},
    "합병증": {"triggers": ["합병증", "부작용"], "keywords": ["합병증"]},
    "관리": {"triggers": ["관리", "생활", "일상", "평소"], "keywords": ["관리", "실천", "생활습관"]},
}
OVERVIEW_INTENT = {"triggers": ["뭐야", "무엇", "이란", "란?", "정의", "설명"], "keywords": ["개요", "정의", "요약"]}


def detect_diseases(text):
    found = []
    remaining = text
    for disease in KNOWN_DISEASES:
        if disease in remaining:
            found.append(disease)
            remaining = remaining.replace(disease, " ", 1)
    return sorted(found, key=lambda d: text.find(d))


def detect_category_keywords(text):
    keywords = []
    for intent in CATEGORY_INTENTS.values():
        if any(trigger in text for trigger in intent["triggers"]):
            keywords.extend(intent["keywords"])
    if keywords:
        return keywords
    if any(trigger in text for trigger in OVERVIEW_INTENT["triggers"]):
        return OVERVIEW_INTENT["keywords"]
    return []


def resolve_categories(disease, category_keywords):
    if not category_keywords:
        return []
    actual_categories = DISEASE_CATEGORIES.get(disease, [])
    return [c for c in actual_categories if any(k in c for k in category_keywords)]


# --- 테스트셋: 병명 언급 8개 + 증상만 언급 7개 ---
DISEASE_INDICES = [0, 2, 3, 5, 7, 11, 16, 23]  # 원인/진단/약물/합병증/개요/치료/예방/관리 골고루

SYMPTOM_TESTS = list(SYMPTOM_CASES) + [
    ("허리가 아파요", ["요통", "강직성척추염", "추간판탈출증(디스크)", "강직척추염", "염좌", "척추관 협착증", "척추의 형태 이상(척추후만증)"]),
    ("머리가 어지럽다", ["어지럼", "실신", "빈혈"]),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_by_disease_category",
            "description": "질문에 병명이 명시된 경우, 그 병의 특정 카테고리(원인/증상/치료/예방/합병증/진단/관리 등) 정보를 검색한다",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease": {"type": "string", "description": "정확한 병명"},
                    "category": {"type": "string", "description": "원인, 증상, 치료, 예방, 합병증, 진단, 관리 중 하나"},
                },
                "required": ["disease", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_symptom",
            "description": "질문에 병명이 없고 증상 설명만 있는 경우, 그 증상으로 관련 질환을 검색한다",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom_text": {"type": "string", "description": "핵심 증상을 정리한 검색용 문구"},
                },
                "required": ["symptom_text"],
            },
        },
    },
]


def rule_based_route_disease(question):
    t0 = time.time()
    diseases = detect_diseases(question)
    keywords = detect_category_keywords(question)
    disease = diseases[0] if diseases else None
    resolved = resolve_categories(disease, keywords) if disease else []
    return disease, resolved, time.time() - t0


def agentic_route(llm, question):
    t0 = time.time()
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "당신은 의료 정보 검색 도우미입니다. 질문에 병명이 있으면 search_by_disease_category를, "
                                                "병명 없이 증상만 있으면 search_by_symptom을 호출하세요."},
                {"role": "user", "content": question},
            ],
            tools=TOOLS,
            tool_choice="auto",
        )
        elapsed = time.time() - t0
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return None, elapsed, "tool 호출 안 함"
        call = tool_calls[0]["function"]
        args = json.loads(call["arguments"])
        return {"name": call["name"], "args": args}, elapsed, None
    except Exception as e:
        return None, time.time() - t0, f"오류: {e}"


def category_matches(predicted, expected_list):
    if not predicted:
        return False
    return any(predicted in e or e in predicted for e in expected_list)


def main_run():
    print("에이전틱 라우팅용 LLM 로딩 (chatml-function-calling)...")
    llm = Llama(
        model_path="models/meta-llama-3.1-8b-instruct-q4_k_m.gguf",
        n_ctx=4096,
        n_threads=14,
        n_threads_batch=14,
        chat_format="chatml-function-calling",
        verbose=False,
    )
    print("임베딩 모델/ChromaDB 로딩...")
    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    chroma_client = chromadb.PersistentClient(path=os.path.join(PROJECT_ROOT, "vector_db"))
    collection = chroma_client.get_collection("medical_knowledge_v2")

    def vector_search_top_disease(text):
        emb = embed_model.encode([text]).tolist()
        res = collection.query(query_embeddings=emb, n_results=1, include=["metadatas"])
        return res["metadatas"][0][0].get("disease") if res["metadatas"][0] else None

    rule_correct = 0
    agentic_correct = 0
    rule_time_total = 0.0
    agentic_time_total = 0.0
    rows = []

    # --- 1. 병명 언급 문항 ---
    for idx in DISEASE_INDICES:
        question, expected_disease, expected_categories = GROUND_TRUTH[idx]

        r_disease, r_categories, r_time = rule_based_route_disease(question)
        r_ok = (r_disease == expected_disease) and category_matches(r_categories[0] if r_categories else None, expected_categories)

        call, a_time, a_error = agentic_route(llm, question)
        if call and call["name"] == "search_by_disease_category":
            a_disease = call["args"].get("disease")
            a_category = call["args"].get("category")
            a_ok = (a_disease == expected_disease) and category_matches(a_category, expected_categories)
        else:
            a_disease, a_category = None, None
            a_ok = False

        rule_correct += r_ok
        agentic_correct += a_ok
        rule_time_total += r_time
        agentic_time_total += a_time
        rows.append({"type": "disease", "question": question, "expected_disease": expected_disease,
                     "expected_categories": expected_categories,
                     "rule": {"disease": r_disease, "categories": r_categories, "time": r_time, "ok": r_ok},
                     "agentic": {"call": call, "time": a_time, "error": a_error, "ok": a_ok}})

        flag = "" if r_ok == a_ok else "  <<< 차이"
        print(f"[병명형] {question}")
        print(f"  규칙기반: disease={r_disease}, category={r_categories} ({r_time*1000:.1f}ms) {'OK' if r_ok else 'FAIL'}")
        print(f"  에이전틱: {call} ({a_time:.1f}s) {'OK' if a_ok else 'FAIL'}{' (' + a_error + ')' if a_error else ''}{flag}")

    # --- 2. 증상만 언급 문항 ---
    for question, acceptable in SYMPTOM_TESTS:
        t0 = time.time()
        r_top = vector_search_top_disease(question)
        r_time = time.time() - t0
        r_ok = r_top in acceptable

        call, a_time, a_error = agentic_route(llm, question)
        if call and call["name"] == "search_by_symptom":
            search_text = call["args"].get("symptom_text", question)
            a_time2 = time.time()
            a_top = vector_search_top_disease(search_text)
            a_time += time.time() - a_time2
            a_ok = a_top in acceptable
        elif call and call["name"] == "search_by_disease_category":
            # 증상 질문인데 억지로 병명을 지어내려 한 경우 (그 자체로 오답 신호)
            a_top = call["args"].get("disease")
            a_ok = a_top in acceptable
        else:
            a_top = None
            a_ok = False

        rule_correct += r_ok
        agentic_correct += a_ok
        rule_time_total += r_time
        agentic_time_total += a_time
        rows.append({"type": "symptom", "question": question, "acceptable": acceptable,
                     "rule": {"top_disease": r_top, "time": r_time, "ok": r_ok},
                     "agentic": {"call": call, "top_disease": a_top, "time": a_time, "error": a_error, "ok": a_ok}})

        flag = "" if r_ok == a_ok else "  <<< 차이"
        print(f"[증상형] {question}")
        print(f"  규칙기반: top_disease={r_top} ({r_time*1000:.1f}ms) {'OK' if r_ok else 'FAIL'}")
        print(f"  에이전틱: {call} -> top_disease={a_top} ({a_time:.1f}s) {'OK' if a_ok else 'FAIL'}{' (' + a_error + ')' if a_error else ''}{flag}")

    total = len(DISEASE_INDICES) + len(SYMPTOM_TESTS)
    print()
    print("=" * 80)
    print(f"규칙 기반 정확도: {rule_correct}/{total} ({rule_correct/total:.1%}), 총 소요시간: {rule_time_total:.2f}s")
    print(f"에이전틱 정확도:  {agentic_correct}/{total} ({agentic_correct/total:.1%}), 총 소요시간: {agentic_time_total:.1f}s")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "rule_correct": rule_correct, "agentic_correct": agentic_correct, "total": total,
            "rule_time_total": rule_time_total, "agentic_time_total": agentic_time_total,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 {out_path}에 저장되었습니다.")


if __name__ == "__main__":
    main_run()
