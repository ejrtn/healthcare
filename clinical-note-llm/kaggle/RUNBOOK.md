# Kaggle 실행 순서 (GPU T4 x2, Internet On)

학습 라이브러리(trl·bitsandbytes)는 vLLM보다 **먼저** 쓴다 — vLLM을 설치하면 torch가 CUDA 13 빌드로 바뀐다.
모든 셀은 `%cd /kaggle/working/healthcare/clinical-note-llm`로 시작한다.

## 2차 학습: 근거 정리 데이터 (`data/sft_grounded`)

정답 기록에서 대화에 없는 나이·성을 지운 데이터로 1차와 같은 설정으로 학습한다. 1차와 비교할 것:
ROUGE(같은 디코딩, frequency_penalty 0.4) + 근거 검사(`src/faithfulness.py`).

```python
# 셀 1: 코드 + 학습 라이브러리
%cd /kaggle/working
!git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git || (cd healthcare && git pull -q)
%cd /kaggle/working/healthcare/clinical-note-llm
!pip install -q trl peft bitsandbytes && pip uninstall -y -q torchao  # 낡은 torchao(0.10)가 있으면 peft 병합이 실패한다
```

```python
# 셀 2: 학습(약 25분) → 병합. 실패하면 병합하지 않도록 &&로 잇는다
%cd /kaggle/working/healthcare/clinical-note-llm
!torchrun --nproc_per_node 2 kaggle/02_sft.py --run-name qwen3b-r16-grounded --data-dir data/sft_grounded && python kaggle/02_sft.py --run-name qwen3b-r16-grounded --merge-only
```

```python
# 셀 3: vLLM 설치 — 끝나면 Run → Restart & clear cell outputs
!pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 4: 테스트 120건 (약 10분)
%cd /kaggle/working/healthcare/clinical-note-llm
!python kaggle/01_baseline.py --model outputs/sft/qwen3b-r16-grounded/merged --mode json --group sft --tag qwen3b-r16-grounded-fp0.4 --frequency-penalty 0.4
!python src/faithfulness.py outputs/baseline/Qwen2.5-3B-Instruct-fp0.4_json.jsonl outputs/sft/qwen3b-r16-fp0.4_json.jsonl outputs/sft/qwen3b-r16-grounded-fp0.4_json.jsonl
```

## 서빙 벤치마크 재측정

1차 측정의 두 문제(같은 대화 재전송으로 prefix cache 적중, 모델 generation_config가 서버 기본값을 덮어씀)를
고친 버전으로 다시 잰다. 속도는 가중치 값과 무관하므로 2차 모델로 잰다.

```python
# 셀 5: 설정마다 10~15분
%cd /kaggle/working/healthcare/clinical-note-llm
M = "outputs/sft/qwen3b-r16-grounded/merged"
!python kaggle/03_serving_bench.py --model {M} --tag v2-fp16-tp1
!python kaggle/03_serving_bench.py --model {M} --tag v2-fp16-tp1-json --json
!python kaggle/03_serving_bench.py --model {M} --tag v2-fp16-tp2 --tp 2
!python kaggle/03_serving_bench.py --model Qwen/Qwen2.5-3B-Instruct-AWQ --tag v2-awq4-base-tp1
```

```python
# 셀 6: 결과 압축 → Output 패널에서 받아 clinical-note-llm/ 에 푼다
%cd /kaggle/working/healthcare/clinical-note-llm
!zip -q -r /kaggle/working/round2.zip outputs/sft/*grounded* results/sft_*grounded* results/faithfulness.json results/serving_v2-* outputs/serving/v2-*.log outputs/sft/qwen3b-r16-grounded/train_stats.json -x "*/merged/*" "*/adapter/*"
!ls -la /kaggle/working/round2.zip
```

병합된 모델(`outputs/sft/qwen3b-r16-grounded/merged`, 약 6GB)은 CPU용 GGUF 변환에 쓴다. 로컬 PC에서
변환하려면 어댑터(`outputs/sft/qwen3b-r16-grounded/adapter`, 약 120MB)만 받아 오면 된다.
