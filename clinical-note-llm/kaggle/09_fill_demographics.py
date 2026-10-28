"""차트의 빈 이름·나이·성별을 대화와 기록 내용에 맞게 채운다 (Kaggle, vLLM).

정답 기록은 "Jacqueline is a pleasant 45-year-old female who presents…"처럼 이름·나이·성별로 시작하는 경우가 많다.
실제 EHR에는 이 세 가지가 항상 있지만, 정답 기록에서 복원한 차트(08_extract_chart.py)는 기록에 없으면 빈칸이 됐고,
빈칸 때문에 모델이 나이를 지어냈다(차트 SFT 1차, 나이 없는 29건 중 16건).

이 세 칸만 채운다. 대화나 기록에 나오면 그 값을 쓰고(예: 대화 "hi, sarah" → Sarah), 없을 때만 내용과 어긋나지 않게
지어낸다(학생이면 20대 초반, 심근경색 병력이면 50대 이상). 약·검사 결과처럼 실제로도 없을 수 있는 칸은 건드리지 않는다.
키워드 규칙으로 해 보니 "고등학교 때 담배를 시작했다"를 고등학생으로 읽는 식의 오류가 나서, 내용을 읽는 일은 LLM에 맡긴다.

    !python kaggle/09_fill_demographics.py

결과: data/chart/extracted/demographics_{split}.jsonl — {"id", "name", "age", "sex", "source": {"name": "stated|invented", ...}}
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle"))

from evaluate import load_jsonl  # noqa: E402

load_cases = __import__("08_extract_chart").load_cases

FILL_PROMPT = """You complete the demographics (name, age, sex) of a patient's chart before a clinic visit.

You get the CONVERSATION, the clinical NOTE written after the visit, and the CURRENT values (empty = missing).
Fill every missing value:
- If the conversation or note states it, use exactly that. A first name the doctor uses to greet the patient
  ("hi, sarah") is the patient's first name; add a plausible surname.
- Otherwise invent a realistic value that agrees with everything in the conversation and note: life situation
  (student, retired, grandchildren, pregnancy), conditions and history (e.g. a prior heart attack or stent suggests
  an older adult), and pronouns used for the patient.
- Never contradict the conversation or note, and never change a value that is already filled.

For each field, say whether the value was "stated" (in the conversation or note) or "invented"."""

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 110},
        "sex": {"type": "string", "enum": ["Female", "Male"]},
        "source": {
            "type": "object",
            "properties": {k: {"type": "string", "enum": ["stated", "invented", "given"]} for k in ["name", "age", "sex"]},
            "required": ["name", "age", "sex"],
            "additionalProperties": False,
        },
    },
    "required": ["name", "age", "sex", "source"],
    "additionalProperties": False,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    try:  # vLLM 버전에 따라 structured output 인터페이스 이름이 다르다
        from vllm.sampling_params import StructuredOutputsParams
        params = SamplingParams(temperature=0, max_tokens=200, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        params = SamplingParams(temperature=0, max_tokens=200, guided_decoding=GuidedDecodingParams(json=SCHEMA))

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="float16", max_model_len=12288,
              gpu_memory_utilization=0.90, seed=0)
    for split, rows in load_cases().items():
        # 걸러낸 차트(build_chart_data.py)의 현재 값. 이미 있는 값은 그대로 두라고 알려 준다
        charts = {r["id"]: r["chart_strict"] for r in load_jsonl(ROOT / "data" / "chart" / f"{split}.jsonl")}
        msgs = []
        for cid, dialogue, note in rows:
            current = {k: charts[cid][k] for k in ["name", "age", "sex"]}
            msgs.append([{"role": "system", "content": FILL_PROMPT},
                         {"role": "user", "content": f"CONVERSATION:\n{dialogue}\n\nNOTE:\n{note}\n\n"
                                                     f"CURRENT:\n{json.dumps(current)}"}])
        outputs = llm.chat(msgs, params)
        failed = 0
        with open(ROOT / "data" / "chart" / "extracted" / f"demographics_{split}.jsonl", "w", encoding="utf-8") as f:
            for (cid, _, _), o in zip(rows, outputs):
                try:
                    d = json.loads(o.outputs[0].text)
                except json.JSONDecodeError:
                    d, failed = None, failed + 1
                f.write(json.dumps({"id": cid} | (d or {}), ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)}건, 실패 {failed}건")


if __name__ == "__main__":
    main()
