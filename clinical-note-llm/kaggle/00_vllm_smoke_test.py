"""Kaggle T4 x2 환경에서 vLLM이 이 프로젝트에 쓸 수 있는 상태인지 확인하는 스모크 테스트.

Kaggle 노트북 설정: Accelerator = GPU T4 x2, Internet = On.
첫 셀:
    !pip install -q vllm
    !pip uninstall -y -q torchaudio   # vllm이 torch를 CUDA 13 빌드로 올리면, Kaggle 기본
                                      # torchaudio(CUDA 12.8)와 충돌해 import 단계에서 죽는다
그다음 커널 재시작(Run → Restart, 설치 패키지는 유지됨) 후 이 파일 내용을 셀에 붙여 넣고 실행한다.

확인 항목 (하나라도 실패하면 GPU 환경 계획을 바꿔야 한다):
  1. vLLM이 T4(sm_75)에서 FP16으로 모델을 띄우는가
  2. ACI-BENCH 대화 1건(약 1,700토큰)을 넣고 진료기록을 생성하는가
  3. JSON 스키마 강제(structured output)가 동작하는가
  4. tensor_parallel_size=2 (T4 두 장)로도 뜨는가
"""
import csv
import io
import json
import subprocess
import time
import urllib.request

MODEL = "Qwen/Qwen2.5-3B-Instruct"
ACI_TRAIN_URL = (
    "https://raw.githubusercontent.com/wyim/aci-bench/"
    "b909b2bb9cf1d19de08df15cddde7bd0179665e4/data/challenge_data/train.csv"
)

NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "subjective": {"type": "string"},
        "objective": {"type": "string"},
        "assessment_and_plan": {"type": "string"},
    },
    "required": ["subjective", "objective", "assessment_and_plan"],
}

SYSTEM = (
    "You are a medical scribe. Read the doctor-patient conversation and write "
    "the clinical note. Only include facts stated in the conversation."
)


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def load_dialogue():
    with urllib.request.urlopen(ACI_TRAIN_URL) as resp:
        rows = list(csv.DictReader(io.StringIO(resp.read().decode("utf-8"))))
    return rows[0]["dialogue"]


def run(tp_size):
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print(f"\n=== vLLM {vllm.__version__}, torch {torch.__version__}, tp={tp_size} ===")
    llm = LLM(
        model=MODEL,
        dtype="float16",  # T4에는 bfloat16 텐서코어가 없다
        tensor_parallel_size=tp_size,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )
    report("모델 로드", True)

    dialogue = load_dialogue()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": dialogue},
    ]
    n_prompt = len(llm.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True))

    start = time.perf_counter()
    out = llm.chat(messages, SamplingParams(temperature=0, max_tokens=800))[0]
    elapsed = time.perf_counter() - start
    n_out = len(out.outputs[0].token_ids)
    report("자유 형식 생성", n_out > 0,
           f"prompt {n_prompt} tok, output {n_out} tok, {elapsed:.1f}s, {n_out / elapsed:.1f} tok/s")
    print(out.outputs[0].text[:500])

    # vLLM 버전에 따라 structured output 인터페이스 이름이 다르다
    try:
        from vllm.sampling_params import StructuredOutputsParams
        params = SamplingParams(temperature=0, max_tokens=1200,
                                structured_outputs=StructuredOutputsParams(json=NOTE_SCHEMA))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        params = SamplingParams(temperature=0, max_tokens=1200,
                                guided_decoding=GuidedDecodingParams(json=NOTE_SCHEMA))
    text = llm.chat(messages, params)[0].outputs[0].text
    try:
        parsed = json.loads(text)
        report("JSON 스키마 강제", set(parsed) == set(NOTE_SCHEMA["required"]), str(list(parsed)))
    except json.JSONDecodeError as e:
        report("JSON 스키마 강제", False, f"파싱 실패: {e}")


if __name__ == "__main__":
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                          "--format=csv"], capture_output=True, text=True).stdout)
    try:
        run(tp_size=1)
    except Exception as e:
        report("tp=1 전체", False, repr(e))
    # tp=2는 프로세스를 새로 띄워야 GPU 메모리가 깨끗하다 — 실패해도 tp=1 결과는 유효
    print("\ntp=2 확인은 노트북을 재시작한 뒤 run(tp_size=2)만 따로 실행할 것.")
