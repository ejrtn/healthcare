"""DPO 3단계: 후보 진료기록을 채점해 좋은 답(chosen)·나쁜 답(rejected) 짝을 만든다. CPU에서 돈다.

후보마다 벌점 = 2 × 대화와 반대(LLM 판정) + 대화에 없는 사실(LLM 판정)
                + 근거 없는 숫자·검사·나이(규칙, src/faithfulness.py)
벌점이 가장 낮은 후보가 좋은 답, 그보다 벌점이 2 이상 높은 후보가 나쁜 답이다.

꼼수 막기: "지어낸 숫자가 있으면 나쁜 답"만 가르치면 모델이 숫자·내용을 아예 빼는 쪽으로 배울 수 있다
(빈 기록은 벌점 0). 그래서 좋은 답이 나쁜 답보다
  - 정답과의 ROUGE-1이 3점 넘게 낮지 않고
  - 대화에 근거가 있는 숫자를 2개 이상 덜 쓰지 않고
  - 15% 넘게 짧지 않을 때만
짝으로 쓴다. 짝은 "덜 쓴 답"이 아니라 "같은 만큼 쓰되 틀리지 않은 답"을 고르게 된다.

    python src/build_dpo_pairs.py

입력: outputs/dpo/candidates_{split}<tag>.jsonl, outputs/judge/candidates_{split}<tag>.jsonl, data/sft_grounded/{split}.jsonl
출력: data/dpo<tag>/{train,valid}.jsonl — {"id", "prompt", "chosen", "rejected", "meta"}, results/dpo_pairs<tag>.json (통계)
"""
import argparse
import json
from pathlib import Path
from statistics import mean

from evaluate import load_jsonl, parse_prediction, rouge
from faithfulness import check

ROOT = Path(__file__).resolve().parent.parent


def score(cand, issues, dialogue, gold_note):
    note, divisions = parse_prediction(cand["prediction"])
    faith = check(note, dialogue)
    usable = divisions is not None and cand["finish_reason"] != "length" and issues is not None
    contra = sum(i["type"] == "contradiction" for i in issues or [])
    unsup = sum(i["type"] == "unsupported" for i in issues or [])
    rule = len(faith["unsupported_numbers"]) + len(faith["unsupported_tests"]) + len(faith["invented_age"])
    return {
        "sample": cand["sample"], "prediction": cand["prediction"], "usable": usable,
        "penalty": 2 * contra + unsup + rule,
        "contradictions": contra, "unsupported": unsup, "rule_flags": rule,
        "supported_numbers": faith["numbers_total"] - len(faith["unsupported_numbers"]),
        "words": len(note.split()),
        "rouge1": rouge(gold_note, note)["rouge1"],
    }


def pick_pairs(scored, args):
    """한 대화의 후보 점수 목록 → [(좋은 답, 나쁜 답), ...]"""
    usable = [s for s in scored if s["usable"]]
    if len(usable) < 2:
        return []
    chosen = min(usable, key=lambda s: (s["penalty"], -s["rouge1"]))
    rejected = [
        s for s in usable
        if s["penalty"] >= chosen["penalty"] + args.min_gap
        and chosen["rouge1"] >= s["rouge1"] - args.rouge_slack
        and chosen["supported_numbers"] >= s["supported_numbers"] - args.number_slack
        and chosen["words"] >= args.length_ratio * s["words"]
    ]
    rejected.sort(key=lambda s: -s["penalty"])
    return [(chosen, r) for r in rejected[:args.max_pairs]]


def build(split, args):
    examples = {r["id"]: r for r in load_jsonl(ROOT / args.data_dir / f"{split}.jsonl")}
    judged = {(j["id"], j["sample"]): j["issues"] for j in load_jsonl(ROOT / "outputs" / "judge" / f"candidates_{split}{args.tag}.jsonl")}
    by_id = {}
    for c in load_jsonl(ROOT / "outputs" / "dpo" / f"candidates_{split}{args.tag}.jsonl"):
        by_id.setdefault(c["id"], []).append(c)

    pairs, no_pair = [], 0
    for cid, cands in by_id.items():
        msgs = examples[cid]["messages"]
        gold_note = parse_prediction(msgs[-1]["content"])[0]
        scored = [score(c, judged.get((cid, c["sample"])), msgs[1]["content"], gold_note) for c in cands]
        found = pick_pairs(scored, args)
        no_pair += not found
        for ch, rj in found:
            meta = {k: {"chosen": ch[k], "rejected": rj[k]} for k in
                    ["sample", "penalty", "contradictions", "unsupported", "rule_flags", "supported_numbers", "words", "rouge1"]}
            pairs.append({"id": cid, "prompt": msgs[:-1],
                          "chosen": [{"role": "assistant", "content": ch["prediction"]}],
                          "rejected": [{"role": "assistant", "content": rj["prediction"]}],
                          "meta": meta})

    out = ROOT / "data" / f"dpo{args.tag}" / f"{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    side = lambda k, w: round(mean(p["meta"][k][w] for p in pairs), 2) if pairs else None  # noqa: E731
    stats = {"dialogues": len(by_id), "pairs": len(pairs), "dialogues_without_pair": no_pair}
    for k in ["penalty", "contradictions", "unsupported", "rule_flags", "supported_numbers", "words", "rouge1"]:
        stats[k] = {"chosen": side(k, "chosen"), "rejected": side(k, "rejected")}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/sft_grounded")
    ap.add_argument("--tag", default="", help="05_dpo_candidates.py에 준 --tag")
    ap.add_argument("--min-gap", type=int, default=2, help="나쁜 답은 좋은 답보다 벌점이 이만큼 높아야 한다")
    ap.add_argument("--max-pairs", type=int, default=2, help="대화당 최대 짝 수")
    ap.add_argument("--rouge-slack", type=float, default=0.03)
    ap.add_argument("--number-slack", type=int, default=1)
    ap.add_argument("--length-ratio", type=float, default=0.85)
    args = ap.parse_args()

    report = {"settings": vars(args)} | {split: build(split, args) for split in ["train", "valid"]}
    (ROOT / "results" / f"dpo_pairs{args.tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
