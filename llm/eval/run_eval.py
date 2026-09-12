"""
RAG 파이프라인 회귀 테스트 — 세 갈래로 나눠서 검증한다.

1. 카테고리 라우팅 (GROUND_TRUTH): 병명이 명시된 질문이 정확한 카테고리로
   필터링되는지. 리랭킹 전(raw)과 후(final)를 둘 다 채점해서 리랭킹이 실제로
   도움이 되는지도 같이 확인한다.
2. 거절(REJECT_CASES): 데이터베이스에 없는/무관한 질문에 "관련 없음"으로
   정직하게 답하는지 (THRESHOLD 게이트가 제대로 막는지).
3. 증상 기반(SYMPTOM_CASES): 병명 없이 증상만 말하는 질문에서 전체 검색
   폴백이 그 증상의 감별진단 후보군(acceptable_diseases) 중 하나를 최상위로
   찾아오는지 채점한다.

실행:
    cd project
    python eval/run_eval.py

주의: main.py를 임포트하면 8B LLM과 리랭커까지 전부 로드된다 (이 테스트엔 LLM이
필요 없지만, main.py가 모듈 레벨에서 즉시 로드하는 구조라 그렇다). 그래서 첫 실행에
수십 초가 걸린다.
"""
import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main  # noqa: E402
from ground_truth import GROUND_TRUTH, REJECT_CASES, SYMPTOM_CASES  # noqa: E402


def run_category_routing():
    raw_correct = 0
    final_correct = 0
    rows = []

    for question, disease, expected in GROUND_TRUTH:
        category_keywords = main.detect_category_keywords(question)
        query_embedding = main.embed_model.encode([question]).tolist()

        docs, distances, metadatas = main.search_disease(query_embedding, disease, category_keywords, 3)

        # RAW/FINAL 둘 다 THRESHOLD를 통과해야 "쓸 수 있는 답"으로 친다. RAW가 이 검사를
        # 안 받으면, 실제로는 거절됐을 답을 정답으로 잘못 세서 리랭킹과 불공정하게 비교된다.
        if not docs or distances[0] > main.THRESHOLD:
            raw_top = None
            raw_ok = False
            final_top = None
            final_ok = False
        else:
            raw_top = metadatas[0].get("category")
            raw_ok = raw_top in expected

            _, _, final_metadatas = main.rerank(question, docs, distances, metadatas, 3)
            final_top = final_metadatas[0].get("category") if final_metadatas else None
            final_ok = final_top in expected

        raw_correct += raw_ok
        final_correct += final_ok
        rows.append({
            "question": question, "disease": disease, "expected": expected,
            "raw_top": raw_top, "raw_ok": raw_ok,
            "final_top": final_top, "final_ok": final_ok,
        })

    total = len(GROUND_TRUTH)
    print("=" * 100)
    print("1. 카테고리 라우팅")
    print("=" * 100)
    print(f"{'질문':<32} {'기대':<22} {'RAW':<18} {'FINAL(리랭킹)':<18}")
    print("-" * 100)
    for r in rows:
        flag = "" if r["raw_ok"] == r["final_ok"] else "  <<< 차이"
        print(f"{r['question']:<32} {'/'.join(r['expected']):<22} "
              f"{str(r['raw_top']) + ('✓' if r['raw_ok'] else '✗'):<18} "
              f"{str(r['final_top']) + ('✓' if r['final_ok'] else '✗'):<18}{flag}")
    print("-" * 100)
    print(f"RAW(리랭킹 전)   정확도: {raw_correct}/{total} ({raw_correct/total:.1%})")
    print(f"FINAL(리랭킹 후) 정확도: {final_correct}/{total} ({final_correct/total:.1%})")
    print()

    return {"total": total, "raw_correct": raw_correct, "final_correct": final_correct, "rows": rows}


def run_reject_cases():
    print("=" * 100)
    print("2. 거절(reject) 테스트 — '관련 없음'으로 정직하게 답해야 하는 질문")
    print("=" * 100)

    correct = 0
    rows = []
    for question in REJECT_CASES:
        diseases = main.detect_diseases(question)
        query_embedding = main.embed_model.encode([question]).tolist()

        if diseases:
            # 무관한 질문인데 우연히 등록된 병명 문자열을 포함하는 경우
            docs, distances, metadatas = main.search_disease(query_embedding, diseases[0], [], 3)
        else:
            docs, distances, metadatas = main.vector_search(query_embedding, n_results=3)

        rejected = (not docs) or (distances[0] > main.THRESHOLD)
        correct += rejected
        top_dist = distances[0] if distances else None
        rows.append({"question": question, "detected_diseases": diseases, "distance": top_dist, "rejected": rejected})
        status = "OK(거절함)" if rejected else "FAIL(답을 만들려 함)"
        dist_str = f"{top_dist:.3f}" if top_dist is not None else "N/A"
        print(f"  {question:<32} 감지병명={diseases} 거리={dist_str} -> {status}")

    total = len(REJECT_CASES)
    print(f"\n거절 정확도: {correct}/{total} ({correct/total:.1%})")
    print()
    return {"total": total, "correct": correct, "rows": rows}


def run_symptom_cases():
    print("=" * 100)
    print("3. 증상 기반(병명 미언급) 질문 — 감별진단 후보군 매칭")
    print("=" * 100)

    correct = 0
    rows = []
    for question, acceptable_diseases in SYMPTOM_CASES:
        # /chat과 동일하게: 질문에 우연히 등록된 병명이 섞여 있으면 그 경로를 타고,
        # 아니면 전체 검색 폴백을 탄다.
        diseases = main.detect_diseases(question)
        query_embedding = main.embed_model.encode([question]).tolist()
        if diseases:
            docs, distances, metadatas = main.search_disease(query_embedding, diseases[0], [], 3)
        else:
            docs, distances, metadatas = main.vector_search(query_embedding, n_results=3)

        rejected = (not docs) or (distances[0] > main.THRESHOLD)
        top = metadatas[0] if (metadatas and not rejected) else None
        top_disease = top.get("disease") if top else None
        ok = (not rejected) and (top_disease in acceptable_diseases)
        correct += ok

        rows.append({
            "question": question,
            "acceptable_diseases": acceptable_diseases,
            "rejected": rejected,
            "top_disease": top_disease,
            "top_category": top.get("category") if top else None,
            "distance": distances[0] if distances else None,
            "ok": ok,
        })
        if rejected:
            print(f"  {question:<36} -> 관련없음으로 거절됨  [FAIL]")
        else:
            flag = "OK" if ok else "FAIL"
            print(f"  {question:<36} -> {top_disease} / {top['category']} (거리={distances[0]:.3f})  [{flag}]")

    total = len(SYMPTOM_CASES)
    print(f"\n증상 기반 정확도: {correct}/{total} ({correct/total:.1%})")
    print()
    return {"total": total, "correct": correct, "rows": rows}


def run():
    category_result = run_category_routing()
    reject_result = run_reject_cases()
    symptom_result = run_symptom_cases()

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category_routing": category_result,
        "reject_cases": reject_result,
        "symptom_cases": symptom_result,
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest_run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"결과가 {out_path}에 저장되었습니다.")


if __name__ == "__main__":
    run()
