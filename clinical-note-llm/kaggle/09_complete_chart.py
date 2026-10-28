"""차트에서 빠진 정보를 채운다: 대화가 가리키는데 차트에 없는 것 (Kaggle, vLLM).

의사는 차트를 보면서 대화한다. "혈압 괜찮네요", "리시노프릴 계속 드시죠"처럼 대화가 가리키는 정보는 차트에 있어야
하는데, 정답 기록에서 복원한 차트(08_extract_chart.py)는 기록에 값이 없으면 비었다. 빈칸이 있으면 모델이 정답 기록의
습관대로 값을 지어냈다(차트 SFT 1차: 차트에 나이가 없는 29건 중 16건에서 나이를 지어냄).

채우는 규칙
  - 이름·나이·성별: 실제 EHR에는 항상 있으므로 항상 채운다.
  - 그 밖의 항목(활력징후·지난 검사·과거력·약·알레르기): 대화가 언급하는데 차트에 값이 없을 때만 채운다.
  - 값은 정답 기록에 있으면 그대로 옮기고("note"), 없으면 대화·기록과 어긋나지 않게 지어낸다("invented").
    예: 대화 "혈압 괜찮네요" + 기록에 수치 없음 → 정상 범위 수치.
  - 대화가 언급하지 않는 항목은 채우지 않는다("not recorded" 유지).
  - 진찰 소견·진단·치료 계획처럼 이번 진료에서 정해지는 것은 넣지 않는다.
LLM이 낸 그대로 보관하고, 걸러내기·합치기는 src/build_chart_data.py가 새 파일로 만든다.

    !python kaggle/09_complete_chart.py

결과: data/chart/extracted/completion_{split}.jsonl
      — {"id", "name", "age", "sex", "source": {...}, "additions": [{"field", "value", "source", "mentioned"}]}
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

FILL_PROMPT = """You complete a patient's chart (EHR) as it existed before a clinic visit. The doctor had this chart open during the visit.

You get the CONVERSATION, the clinical NOTE written after the visit, and the CURRENT chart.

1. Demographics (name, age, sex) — always fill a missing value:
   - If the conversation or note states it, use exactly that. A first name the doctor uses to greet the patient
     ("hi, sarah") is the patient's first name; add a plausible surname.
   - Otherwise invent a realistic value that agrees with the conversation and note (life situation such as student,
     retired or pregnant; conditions and history; pronouns).

2. Other chart items — add an item only when the CONVERSATION refers to it but the CURRENT chart lacks it:
   vital signs, earlier test results, past medical/surgical/social/family history, current medications and doses, allergies.
   Examples: the doctor says "your blood pressure looks good" but the chart has no blood pressure; the patient says
   "i'm still taking the lisinopril" but the chart has no lisinopril or no dose.
   - If the NOTE gives the value, copy it exactly (source "note").
   - Otherwise invent a realistic value that agrees with what was said (source "invented"); "looks good" means a normal value.
   - For "mentioned", quote the short phrase from the conversation that refers to it.
   - Do not add anything the conversation does not refer to.
   - Never add findings from today's exam, diagnoses or assessments made today, the plan, new prescriptions or dose changes.

Never change a value that is already in the CURRENT chart, and never contradict the conversation or note."""

FIELDS = ["vitals", "prior_results", "history", "medications", "allergies"]
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
        "additions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": FIELDS},
                "value": {"type": "string"},
                "source": {"type": "string", "enum": ["note", "invented"]},
                "mentioned": {"type": "string"},
            },
            "required": ["field", "value", "source", "mentioned"],
            "additionalProperties": False,
        }},
    },
    "required": ["name", "age", "sex", "source", "additions"],
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
        params = SamplingParams(temperature=0, max_tokens=1024, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        params = SamplingParams(temperature=0, max_tokens=1024, guided_decoding=GuidedDecodingParams(json=SCHEMA))

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="float16", max_model_len=12288,
              gpu_memory_utilization=0.90, seed=0)
    out_dir = ROOT / "data" / "chart" / "extracted"
    for split, rows in load_cases().items():
        # 걸러낸 차트(build_chart_data.py)의 채우기 전 상태. 이미 있는 값은 그대로 두라고 알려 준다
        charts = {r["id"]: r["chart_text_strict"] for r in load_jsonl(ROOT / "data" / "chart" / f"{split}.jsonl")}
        msgs = [[{"role": "system", "content": FILL_PROMPT},
                 {"role": "user", "content": f"CONVERSATION:\n{dialogue}\n\nNOTE:\n{note}\n\nCURRENT CHART:\n{charts[cid]}"}]
                for cid, dialogue, note in rows]
        outputs = llm.chat(msgs, params)
        failed = 0
        with open(out_dir / f"completion_{split}.jsonl", "w", encoding="utf-8") as f:
            for (cid, _, _), o in zip(rows, outputs):
                try:
                    d = json.loads(o.outputs[0].text)
                except json.JSONDecodeError:
                    d, failed = None, failed + 1
                f.write(json.dumps({"id": cid} | (d or {}), ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)}건, 실패 {failed}건")


if __name__ == "__main__":
    main()
