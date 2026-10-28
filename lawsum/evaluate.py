"""AI 요약, Lead-3 베이스라인, 그리고 "공식 판시사항 vs 공식 판결요지"(둘 다
법제처가 쓴 것 — 사실상 품질 상한선 참고치)까지 세 가지를 공식 판결요지와
비교한다. AI 요약 혼자 숫자(0.477)만 보면 좋은지 나쁜지 알 수 없는데,
베이스라인·상한선과 나란히 놓으면 "LLM이 실제로 얼마나 보탰는지"가 보인다.
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from summarize import strip_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarized", default="data/summarized.json",
                    help="요약 결과 (Kaggle GPU 모델 비교: data/summarized_<모델>.json)")
    ap.add_argument("--out", default="data/evaluated.json")
    ap.add_argument("--baseline", default="data/baseline.json", help="같은 판례로 만든 Lead-3 결과")
    args = ap.parse_args()
    with open(args.summarized, encoding="utf-8") as f:
        ai_cases = json.load(f)
    with open(args.baseline, encoding="utf-8") as f:
        baseline_by_id = {b["_판례일련번호"]: b["baseline_summary"] for b in json.load(f)}

    model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    official = [c["판결요지_clean"] for c in ai_cases]
    ai_summaries = [c["ai_summary"] for c in ai_cases]
    baselines = [baseline_by_id.get(c["_판례일련번호"], "") for c in ai_cases]
    headnotes = [strip_html(c.get("판시사항")) for c in ai_cases]

    emb_official = model.encode(official, convert_to_tensor=True)
    emb_ai = model.encode(ai_summaries, convert_to_tensor=True)
    emb_baseline = model.encode(baselines, convert_to_tensor=True)
    emb_headnote = model.encode(headnotes, convert_to_tensor=True)

    ai_scores, baseline_scores, headnote_scores = [], [], []
    for i, case in enumerate(ai_cases):
        ai_sim = cos_sim(emb_ai[i], emb_official[i]).item()
        base_sim = cos_sim(emb_baseline[i], emb_official[i]).item()
        head_sim = cos_sim(emb_headnote[i], emb_official[i]).item()

        case["similarity_ai_vs_official"] = round(ai_sim, 4)
        case["similarity_baseline_vs_official"] = round(base_sim, 4)
        case["similarity_headnote_vs_official"] = round(head_sim, 4)

        ai_scores.append(ai_sim)
        baseline_scores.append(base_sim)
        headnote_scores.append(head_sim)

        print(f"{case.get('사건명','')[:35]:35s} AI={ai_sim:.3f}  베이스라인={base_sim:.3f}  상한참고={head_sim:.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ai_cases, f, ensure_ascii=False, indent=2)

    def stats(name, scores):
        s = sorted(scores)
        n = len(s)
        print(f"{name}: 평균 {sum(s)/n:.3f}  중앙값 {s[n//2]:.3f}  최소 {s[0]:.3f}  최대 {s[-1]:.3f}")

    print(f"\n=== 전체 {len(ai_cases)}건 요약 ===")
    stats("AI 요약        ", ai_scores)
    stats("Lead-3 베이스라인", baseline_scores)
    stats("공식 판시사항(상한 참고)", headnote_scores)

    improvement = (sum(ai_scores) - sum(baseline_scores)) / len(ai_scores)
    print(f"\nAI 요약이 베이스라인보다 평균 {improvement:+.3f} (코사인 유사도 기준)")


if __name__ == "__main__":
    main()
