"""근거 검색 방식 비교 (CPU): BM25 / 임베딩 / hybrid.

정답 근거 발화 라벨이 없으므로, 근거가 확실한 경우로 평가셋을 만든다. 모델 출력 문장 속 숫자 중 대화에
나오는 숫자(차트에는 없는 것)를 고르고, 그 숫자를 지운 문장을 질의로 쓴다 — "이 숫자는 무엇에 대한 것인가"를
찾는 상황이다. 상위 k개 발화 중 하나에 그 숫자가 있으면(숫자 단어 포함) 적중으로 센다.

    python src/eval_retrieval.py outputs/sft/qwen3b-r16-chart4-fp0.4_json.jsonl
출력: results/retrieval_eval.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import load_jsonl, parse_prediction  # noqa: E402
from evidence import RANKERS, split_turns  # noqa: E402
from faithfulness import TEST_SPLITS, spoken_numbers  # noqa: E402

SENT = re.compile(r"(?:[^.\n]|\.(?=\d))+(?:\.|\n|$)")  # 소수점(12.5)은 문장 끝이 아니다


def build_queries(preds, cases, charts):
    out = []
    for c in cases:
        note, _ = parse_prediction(preds.get(c["id"], ""))
        d_nums, ch_nums = spoken_numbers(c["dialogue"]), spoken_numbers(charts[c["id"]])
        for sent in SENT.findall(note):
            for m in re.finditer(r"\b\d+\b", sent):
                n = int(m.group())
                if n > 3 and n in d_nums and n not in ch_nums:
                    query = (sent[:m.start()] + sent[m.end():]).strip()
                    if len(query.split()) >= 3:
                        out.append({"id": c["id"], "number": n, "query": query})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--chart-dir", default="data/chart")
    ap.add_argument("--out", default=str(ROOT / "results" / "retrieval_eval.json"))
    args = ap.parse_args()

    cases = [g for s in TEST_SPLITS for g in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")]
    charts = {r["id"]: r["chart_text"] for s in TEST_SPLITS for r in load_jsonl(ROOT / args.chart_dir / f"{s}.jsonl")}
    preds = {p["id"]: p["prediction"] for p in load_jsonl(ROOT / args.preds)}
    queries = build_queries(preds, cases, charts)
    turns = {c["id"]: split_turns(c["dialogue"]) for c in cases}

    report = {"queries": len(queries), "cases": len({q["id"] for q in queries}), "methods": {}}
    for name, rank in RANKERS.items():
        start = time.perf_counter()
        hits = {1: 0, 3: 0, 5: 0}
        for q in queries:
            ranking = rank(q["query"], turns[q["id"]])
            found = [q["number"] in spoken_numbers(turns[q["id"]][i]["text"]) for i in ranking[:5]]
            for k in hits:
                hits[k] += any(found[:k])
        report["methods"][name] = {f"hit@{k}": round(100 * v / len(queries), 1) for k, v in hits.items()}
        report["methods"][name]["sec"] = round(time.perf_counter() - start, 1)
        print(name, report["methods"][name], flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"질의 {len(queries)}개 (기록 {report['cases']}건)")


if __name__ == "__main__":
    main()
