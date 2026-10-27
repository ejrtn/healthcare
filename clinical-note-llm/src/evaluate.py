"""생성된 진료기록 채점.

ACI-BENCH 공식 평가(evaluation/evaluate_fullnote.py)와 같은 방식:
  - 전체 기록 + 4구역 각각에 대해 ROUGE-1/2/L/Lsum (rouge_score, stemmer 없음)
  - 구역은 SectionTagger로 나누고, 없는 구역은 '#####EMPTY#####'로 채운다
BERTScore/BLEURT/UMLS 기반 지표는 GPU·UMLS 라이선스가 필요해 여기서는 제외했다.

추가로 이 프로젝트 고유 지표:
  - json_valid: 예측이 4구역 JSON으로 파싱되는 비율 (structured output 비교용)

--lenient: 마크다운 강조(**제목:**, # 제목)를 지우고 채점한다. 자유 텍스트 기준선이
"**ASSESSMENT AND PLAN:**"처럼 써서 구역 분리기가 제목을 못 알아보는 경우, 형식 실패와
내용 품질을 나눠 보기 위한 참고용이다. 공식 점수는 기본(엄격) 모드다.

사용법:
  python src/evaluate.py data/processed/aci_test1.jsonl outputs/preds.jsonl [--out result.json] [--lenient]
  예측 파일: {"id":..., "prediction": "<진료기록 텍스트 또는 4구역 JSON 문자열>"} 한 줄씩
"""
import argparse
import json
import re
from statistics import mean

from rouge_score import rouge_scorer

from schema import DIVISIONS, render_note, split_note

EMPTY = "#####EMPTY#####"
METRICS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
_scorer = rouge_scorer.RougeScorer(METRICS)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_prediction(pred: str):
    """(진료기록 텍스트, 4구역 dict 또는 None).

    4구역 JSON이면 구역은 모델이 키로 직접 나눈 그대로 쓴다. 합친 텍스트를 SectionTagger로
    다시 나누면, 모델이 값 안에 섹션 제목을 빼먹은 경우(3B 기준선에서 112건 중 75건) 구역이
    앞 구역에 붙어 버려 내용과 무관하게 점수가 깎인다. JSON이 아니면 원문을 기록으로 본다.
    """
    try:
        obj = json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        return str(pred), None
    if isinstance(obj, dict) and set(DIVISIONS) <= set(obj):
        divisions = {d: str(obj[d]).strip() for d in DIVISIONS}
        return render_note(divisions), divisions
    return str(pred), None


def strip_markdown(note: str) -> str:
    note = note.replace("**", "")
    return re.sub(r"^\s*#+\s*", "", note, flags=re.M)


def rouge(ref: str, pred: str) -> dict:
    scores = _scorer.score(ref, pred)
    return {m: scores[m].fmeasure for m in METRICS}


def evaluate(golds, preds_by_id, lenient=False):
    per_case, json_ok = [], []
    for g in golds:
        note, json_div = parse_prediction(preds_by_id[g["id"]])
        if lenient:
            note = strip_markdown(note)
        json_ok.append(json_div is not None)
        pred_div = json_div if json_div is not None else split_note(note)
        case = {"id": g["id"], "ALL": rouge(g["note"], note)}
        for d in DIVISIONS:
            case[d] = rouge(g["target"][d] or EMPTY, pred_div[d] or EMPTY)
        per_case.append(case)

    summary = {"num_test": len(golds), "json_valid": mean(json_ok)}
    for key in ["ALL"] + DIVISIONS:
        summary[key] = {m: round(100 * mean(c[key][m] for c in per_case), 2) for m in METRICS}
    return summary, per_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--out")
    ap.add_argument("--lenient", action="store_true")
    args = ap.parse_args()

    golds = load_jsonl(args.gold)
    preds = {p["id"]: p["prediction"] for p in load_jsonl(args.pred)}
    missing = [g["id"] for g in golds if g["id"] not in preds]
    assert not missing, f"예측 누락: {missing[:5]} 외 {len(missing)}건"

    summary, per_case = evaluate(golds, preds, lenient=args.lenient)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_case": per_case}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
