"""LLM 없이 "판례내용 맨 앞 문장 몇 개를 그대로 자른다"(Lead-3) 베이스라인.
AI 요약이 공식 판결요지와 유사도 0.477이 나왔을 때, 이게 "잘한 건지"를
판단하려면 비교 기준이 있어야 한다 — 이 베이스라인이 그 기준이다.
LLM 비용 없이 요약 효과가 얼마나 되는지 수치로 보여준다."""
import json
import re

from summarize import strip_html


def lead_n_sentences(text, n=3):
    text = strip_html(text)
    # 한국어 판결문 문장 종결 패턴(다./함./음. 등 + 마침표) 기준으로 대충 split
    sentences = re.split(r"(?<=[다함음임]\.)\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:n])


def main():
    with open("data/cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        content = case.get("판례내용")
        if not (strip_html(content) or "").strip():
            continue
        baseline_summary = lead_n_sentences(content, n=3)
        results.append({
            "_판례일련번호": case["_판례일련번호"],
            "baseline_summary": baseline_summary,
        })

    with open("data/baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"완료: {len(results)}건 -> data/baseline.json (LLM 호출 없음, 수 초 내 완료)")


if __name__ == "__main__":
    main()
