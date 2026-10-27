"""LLM 판정: 진료기록이 대화와 반대로 쓴 곳(contradiction)과 대화에 없는 사실(unsupported)을 찾는다.

src/faithfulness.py는 숫자·검사 이름·나이처럼 규칙으로 확인되는 것만 센다. "환자가 실금이 한 번 있었다고
했는데 기록에는 denies incontinence"처럼 말뜻을 봐야 하는 오류는 규칙으로 못 잡아서, 더 큰 모델에게
대화와 기록을 나란히 주고 문제 문장을 근거와 함께 JSON으로 뽑게 한다.

쓰임
  1. DPO 짝 고르기: 05_dpo_candidates.py가 만든 후보 채점
  2. 평가 지표: 테스트 예측의 "사실과 반대" 건수를 학습 전·SFT·DPO에 같은 판정기로 센다
판정 모델(Qwen2.5-14B AWQ)은 생성 모델(3B)과 같은 계열이라 같은 실수를 놓칠 수 있다 — 사람이 직접 대조한
10건(README)과 판정 결과를 비교해 얼마나 믿을지 확인한다.

    !python kaggle/06_judge.py outputs/dpo/candidates_train.jsonl outputs/sft/qwen3b-r16-grounded-fp0.4_json.jsonl

결과: outputs/judge/<입력 이름>.jsonl (기록별 문제 목록), results/judge_<입력 이름>.json (요약)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import load_jsonl, parse_prediction  # noqa: E402

SPLITS = ["train", "valid", "test1", "test2", "test3"]

JUDGE_PROMPT = """You check clinical notes against the doctor-patient conversation they were written from.

List every statement in the note that has one of these problems:
- "contradiction": the note says the opposite of the conversation. Examples: the note says the patient denies a symptom the patient reported; wrong side (left/right); a different dose, frequency or duration; a test called normal that the doctor called abnormal.
- "unsupported": the note states a specific fact the conversation never mentions. Examples: an exam finding, test result, medication, allergy, past history, age or plan item that nobody said.

Do not list:
- rephrasing, summaries, or medical terms for something said in plain words
- section headings, or text saying a section has no information
- exam findings or results the doctor dictated during the conversation (they are stated)

For each problem give the exact note text and the conversation evidence (a short quote, or "not mentioned").
Return {"issues": []} if the note has no problems."""

ISSUES_SCHEMA = {
    "type": "object",
    "properties": {"issues": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["contradiction", "unsupported"]},
            "note_text": {"type": "string"},
            "evidence": {"type": "string"},
        },
        "required": ["type", "note_text", "evidence"],
        "additionalProperties": False,
    }}},
    "required": ["issues"],
    "additionalProperties": False,
}


def judge_messages(dialogue, note):
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"CONVERSATION:\n{dialogue}\n\nNOTE:\n{note}"},
    ]


def parse_issues(text):
    """판정 결과 → 문제 목록. 판정 출력이 잘려 JSON이 깨지면 None (집계에서 따로 센다)."""
    try:
        issues = json.loads(text)["issues"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return [i for i in issues if isinstance(i, dict) and i.get("type") in ("contradiction", "unsupported")]


def summarize(rows):
    ok = [r for r in rows if r["issues"] is not None]
    count = lambda r, t: sum(i["type"] == t for i in r["issues"])  # noqa: E731
    n = max(1, len(ok))
    return {
        "notes": len(rows),
        "judge_failed": len(rows) - len(ok),
        "contradictions": sum(count(r, "contradiction") for r in ok),
        "unsupported": sum(count(r, "unsupported") for r in ok),
        "notes_with_contradiction": sum(count(r, "contradiction") > 0 for r in ok),
        "notes_with_unsupported": sum(count(r, "unsupported") > 0 for r in ok),
        "contradictions_per_note": round(sum(count(r, "contradiction") for r in ok) / n, 2),
        "unsupported_per_note": round(sum(count(r, "unsupported") for r in ok) / n, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help='예측 jsonl ({"id", "prediction"}, 후보 파일이면 "sample"도)')
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1536)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    try:  # vLLM 버전에 따라 structured output 인터페이스 이름이 다르다
        from vllm.sampling_params import StructuredOutputsParams
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                                structured_outputs=StructuredOutputsParams(json=ISSUES_SCHEMA))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                                guided_decoding=GuidedDecodingParams(json=ISSUES_SCHEMA))

    dialogues = {c["id"]: c["dialogue"] for s in SPLITS
                 for c in load_jsonl(ROOT / "data" / "processed" / f"aci_{s}.jsonl")}
    # 학습 데이터에는 음성 인식 대화(VS..._asr)도 있다 — 학습에 쓴 대화 그대로 판정한다
    for s in ["train", "valid"]:
        dialogues |= {r["id"]: r["messages"][1]["content"]
                      for r in load_jsonl(ROOT / "data" / "sft_grounded" / f"{s}.jsonl")}
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="float16", max_model_len=12288,
              gpu_memory_utilization=0.90, seed=0)
    (ROOT / "outputs" / "judge").mkdir(parents=True, exist_ok=True)
    for path in args.preds:
        preds = load_jsonl(ROOT / path)
        notes = [parse_prediction(p["prediction"])[0] for p in preds]
        outputs = llm.chat([judge_messages(dialogues[p["id"]], n) for p, n in zip(preds, notes)], params)
        rows = [{"id": p["id"], "sample": p.get("sample"), "issues": parse_issues(o.outputs[0].text)}
                for p, o in zip(preds, outputs)]
        stem = Path(path).stem
        with open(ROOT / "outputs" / "judge" / f"{stem}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize(rows)
        (ROOT / "results" / f"judge_{stem}.json").write_text(
            json.dumps({"judge_model": args.model, "preds": path, "summary": s}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(stem, json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
