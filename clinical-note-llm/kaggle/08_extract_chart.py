"""정답 진료기록에서 거꾸로 "진료 전 차트"를 되살린다.

정답 기록에는 대화에 나오지 않는 나이·이름·혈압·지난 검사 수치·날짜가 있다. 기록 작성자가 진료 정보 시스템(EHR)을
보고 채운 것이다. 모델은 대화만 받으므로 이것을 학습하면 "그럴듯한 수치를 지어내는 법"을 배운다(README 근거 검사).
실제 진료처럼 입력을 "차트 + 대화"로 바꾸기 위해, ACI-BENCH에 없는 차트를 정답 기록에서 복원한다.

차트에 넣는 것: 진료 전부터 시스템에 있었을 정보(이름·나이·성별, 간호사가 잰 활력징후, 지난 검사 결과,
과거력, 복용 약, 알레르기). 대화에 나왔는지와 상관없이 넣는다 — 실제 차트는 대화와 무관하게 늘 같은 내용을 담고,
의사는 그 차트를 보면서 대화한다. 대화에 나온 것만 빼면 차트가 사례마다 들쭉날쭉해져 모델이 차트의 역할을 잘못 배운다.
넣지 않는 것: 이번 진료에서 의사가 판단·결정한 것(진찰 소견, 진단, 치료 계획, 새 처방·용량 변경) — 이것까지 넣으면
모델이 써야 할 답을 입력으로 미리 보게 된다. 결과는 src/build_chart_data.py가 한 번 더 걸러낸다.

    !python kaggle/08_extract_chart.py

결과: data/chart/extracted/chart_{split}.jsonl — {"id", "chart": {...} 또는 null}
(LLM이 낸 그대로 보관하고 고치지 않는다. 걸러내기·정리는 src/build_chart_data.py가 새 파일로 만든다)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import load_jsonl, parse_prediction  # noqa: E402

EXTRACT_PROMPT = """You rebuild the patient's chart (EHR) as it existed before a clinic visit.

You get the doctor-patient CONVERSATION and the clinical NOTE written after the visit. The doctor had the chart open during the visit, and the note writer used it too.

Include a fact if BOTH are true:
1. It appears in the NOTE.
2. It would already be in the chart before the doctor saw the patient: patient name, age, sex, vital signs measured by staff, earlier test results and their dates, past medical/surgical/social/family history, current medications and doses, allergies, dates of earlier visits or procedures.

Include it whether or not the conversation also mentions it — a real chart holds these facts regardless of what is said in the visit.

Never include anything decided or found during this visit: physical exam findings, diagnoses or assessments made today, the treatment plan, new prescriptions, dose changes, referrals, or follow-up instructions.

Copy values exactly as written in the note (numbers, units, dates). Use "" or [] when there is nothing."""

CHART_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "string"},
        "sex": {"type": "string"},
        "vitals": {"type": "array", "items": {"type": "string"}},
        "prior_results": {"type": "array", "items": {"type": "string"}},
        "history": {"type": "array", "items": {"type": "string"}},
        "medications": {"type": "array", "items": {"type": "string"}},
        "allergies": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "sex", "vitals", "prior_results", "history", "medications", "allergies"],
    "additionalProperties": False,
}


def load_cases():
    """분할별 (id, 대화, 정답 기록). 학습·검증은 SFT 데이터(음성 인식 대화 포함), 테스트는 원본."""
    cases = {}
    for split in ["train", "valid"]:
        cases[split] = [(r["id"], r["messages"][1]["content"], parse_prediction(r["messages"][-1]["content"])[0])
                        for r in load_jsonl(ROOT / "data" / "sft" / f"{split}.jsonl")]
    for split in ["test1", "test2", "test3"]:
        cases[split] = [(c["id"], c["dialogue"], c["note"])
                        for c in load_jsonl(ROOT / "data" / "processed" / f"aci_{split}.jsonl")]
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    try:  # vLLM 버전에 따라 structured output 인터페이스 이름이 다르다
        from vllm.sampling_params import StructuredOutputsParams
        params = SamplingParams(temperature=0, max_tokens=1024,
                                structured_outputs=StructuredOutputsParams(json=CHART_SCHEMA))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        params = SamplingParams(temperature=0, max_tokens=1024, guided_decoding=GuidedDecodingParams(json=CHART_SCHEMA))

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="float16", max_model_len=12288,
              gpu_memory_utilization=0.90, seed=0)
    out_dir = ROOT / "data" / "chart" / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in load_cases().items():
        msgs = [[{"role": "system", "content": EXTRACT_PROMPT},
                 {"role": "user", "content": f"CONVERSATION:\n{d}\n\nNOTE:\n{n}"}] for _, d, n in rows]
        outputs = llm.chat(msgs, params)
        failed = 0
        with open(out_dir / f"chart_{split}.jsonl", "w", encoding="utf-8") as f:
            for (cid, _, _), o in zip(rows, outputs):
                try:
                    chart = json.loads(o.outputs[0].text)
                except json.JSONDecodeError:
                    chart, failed = None, failed + 1
                f.write(json.dumps({"id": cid, "chart": chart}, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)}건, 실패 {failed}건")


if __name__ == "__main__":
    main()
