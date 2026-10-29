"""규칙 보정 전후 비교 (CPU, src/correct.py).

    python src/eval_correction.py outputs/sft/qwen3b-r16-chart4-fp0.4_json.jsonl [--chart-field chart_text_strict]

보는 것
  - 근거 검사(v2): 근거 없는 숫자·검사·나이가 얼마나 줄었나
  - 맞는 숫자: 근거 있는 숫자가 지워지지 않았나 (보정은 표시된 값만 지워야 한다)
  - ROUGE-1: 정답 기록과의 겹침이 크게 떨어지지 않았나
  - 정답 라벨(data/flag_labels.json): 실제로 근거 없는 문장은 고쳐지고, 근거 있는 문장은 그대로인가
출력: results/correction_<이름>.json, outputs/corrected/<이름>_json.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from correct import correct_note  # noqa: E402
from evaluate import evaluate, load_jsonl, parse_prediction  # noqa: E402
from faithfulness import TEST_SPLITS, check, summarize  # noqa: E402


def label_scores(preds_path, before, after, source):
    path = ROOT / "data" / "flag_labels.json"
    labels = [x for x in json.loads(path.read_text(encoding="utf-8"))["items"] if x["preds"] == preds_path]
    if not labels:
        return None
    rows = []
    for x in labels:
        b, _ = parse_prediction(before[x["id"]])
        a, _ = parse_prediction(after[x["id"]])
        # 같은 말로 시작하는 문장이 여럿일 수 있어("- Medical Reasoning: ..."), 근거 검사 v1이 표시한 숫자가 든 문장을 고른다
        cands = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", b) if s.strip().startswith(x["sentence_start"])]
        flagged = [s for s in cands if re.search(r"\d|\bct\b|echo", s, re.I)] or cands
        changed = bool(flagged) and flagged[0] not in a
        rows.append({"id": x["id"], "label": x["label"], "changed": changed,
                     "correct": changed if x["label"] == "tp" else not changed})
    tp = [r for r in rows if r["label"] == "tp"]
    fp = [r for r in rows if r["label"] == "fp"]
    return {"sentences": len(rows), "accuracy": round(100 * sum(r["correct"] for r in rows) / len(rows), 1),
            "tp_corrected": f"{sum(r['correct'] for r in tp)}/{len(tp)}",
            "fp_untouched": f"{sum(r['correct'] for r in fp)}/{len(fp)}", "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--chart-dir", default="data/chart")
    ap.add_argument("--chart-field", default="chart_text")
    args = ap.parse_args()

    cases = [g for s in TEST_SPLITS for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]
    charts = {r["id"]: r[args.chart_field] for s in TEST_SPLITS for r in load_jsonl(ROOT / args.chart_dir / f"{s}.jsonl")}
    before = {p["id"]: p["prediction"] for p in load_jsonl(ROOT / args.preds)}
    source = {c["id"]: charts[c["id"]] + "\n\n" + c["dialogue"] for c in cases}

    after, edits, left = {}, [], []
    for c in cases:
        _, divisions = parse_prediction(before[c["id"]])
        if divisions is None:  # JSON이 깨진 출력은 그대로 둔다
            after[c["id"]] = before[c["id"]]
            continue
        fixed, e, l = correct_note(divisions, source[c["id"]])
        after[c["id"]] = json.dumps(fixed, ensure_ascii=False)
        edits += [x | {"id": c["id"]} for x in e]
        if any(l.values()):
            left.append({"id": c["id"], **l})

    def faith(p):
        s = summarize([check(parse_prediction(p[c["id"]])[0], source[c["id"]]) for c in cases])
        return s | {"supported_numbers": s["numbers_total"] - s["unsupported_numbers"]}

    name = Path(args.preds).stem.removesuffix("_json")
    out_dir = ROOT / "outputs" / "corrected"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_json.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps({"id": c["id"], "prediction": after[c["id"]]}, ensure_ascii=False) + "\n")
    rouge_b, _ = evaluate(cases, before)
    rouge_a, _ = evaluate(cases, after)
    report = {
        "preds": args.preds, "chart_field": args.chart_field,
        "edits": len(edits), "edits_by_kind": {k: sum(e["kind"] == k for e in edits) for k in ("number", "age")},
        "faithfulness_before": faith(before), "faithfulness_after": faith(after),
        "rouge1_before": rouge_b["ALL"]["rouge1"], "rouge1_after": rouge_a["ALL"]["rouge1"],
        "rouge1_by_division_after": {k: v["rouge1"] for k, v in rouge_a.items() if isinstance(v, dict) and "rouge1" in v},
        "labels": label_scores(args.preds, before, after, source),
        "left_for_review": left, "edit_list": edits,
    }
    (ROOT / "results" / f"correction_{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
    show = {k: v for k, v in report.items() if k not in ("edit_list", "left_for_review", "rouge1_by_division_after")}
    if show["labels"]:
        show["labels"] = {k: v for k, v in show["labels"].items() if k != "rows"}
    print(json.dumps(show, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
