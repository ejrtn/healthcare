# Kaggle 실행 순서 (GPU T4 x2, Internet On)

학습 라이브러리(trl·bitsandbytes)는 vLLM보다 **먼저** 쓴다 — vLLM을 설치하면 torch가 CUDA 13 빌드로 바뀐다.
모든 셀은 `%cd /kaggle/working/healthcare/clinical-note-llm`로 시작한다.

## 4차: 차트 + 대화로 학습

정답 기록에 있는 대화 밖 정보(이름·나이·혈압·지난 검사 수치·날짜)는 기록 작성자가 차트(EHR)를 보고 쓴 것이다.
ACI-BENCH에는 차트가 없으므로 정답 기록에서 거꾸로 복원해, 입력을 실제 진료처럼 "차트 + 대화"로 바꿔 학습한다.
복원한 차트에 답(진단·계획·진찰 소견)이 섞이지 않았는지 확인하고 학습해야 하므로 **두 번에 나눠** 돌린다.

### 4-A: 차트 복원 (약 30분, 학습 없음) — 대화에 나온 정보도 차트에 넣는 v2

```python
# 셀 1: 코드 + vLLM
%cd /kaggle/working
!git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git || (cd healthcare && git pull -q)
%cd /kaggle/working/healthcare/clinical-note-llm
!pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 2: 14B 모델로 학습·검증·테스트 219건의 차트 복원 → 압축
%cd /kaggle/working/healthcare/clinical-note-llm
!python kaggle/08_extract_chart.py
!zip -q -r /kaggle/working/round4a.zip data/chart/extracted
!ls -la /kaggle/working/round4a.zip
```

`round4a.zip`을 받아 `clinical-note-llm/`에 풀면, 로컬에서 걸러내기(`src/build_chart_data.py`)와 직접 대조 검사를 한 뒤
학습 데이터(`data/sft_chart`)를 GitHub에 올린다. 그다음 4-B.

### 4-B: 학습·평가 (약 1시간 반, 새 세션)

```python
# 셀 1: 코드 + 학습 라이브러리
%cd /kaggle/working
!git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git || (cd healthcare && git pull -q)
%cd /kaggle/working/healthcare/clinical-note-llm
!pip install -q trl peft bitsandbytes && pip uninstall -y -q torchao
```

```python
# 셀 2: 차트 + 대화로 SFT (약 25분) → 병합. 차트가 붙어 입력이 길어져 max-length를 늘린다
%cd /kaggle/working/healthcare/clinical-note-llm
!torchrun --nproc_per_node 2 kaggle/02_sft.py --run-name qwen3b-r16-chart --data-dir data/sft_chart --max-length 5200 && python kaggle/02_sft.py --run-name qwen3b-r16-chart --merge-only
```

```python
# 셀 3: vLLM 설치 — 끝나면 Run → Restart & clear cell outputs
!pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 4: 테스트 120건. 학습 전 모델에도 같은 차트를 줘서 "차트를 준 효과"와 "학습 효과"를 나눠 본다
%cd /kaggle/working/healthcare/clinical-note-llm
!python kaggle/01_baseline.py --model Qwen/Qwen2.5-3B-Instruct --mode json --tag Qwen2.5-3B-Instruct-chart-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart
!python kaggle/01_baseline.py --model outputs/sft/qwen3b-r16-chart/merged --mode json --group sft --tag qwen3b-r16-chart-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart
!python src/faithfulness.py outputs/baseline/Qwen2.5-3B-Instruct-chart-fp0.4_json.jsonl outputs/sft/qwen3b-r16-chart-fp0.4_json.jsonl --chart-dir data/chart --out results/faithfulness_chart.json
```

```python
# 셀 5: 결과 압축 → 받아서 clinical-note-llm/ 에 푼다
%cd /kaggle/working/healthcare/clinical-note-llm
!zip -q -r /kaggle/working/round4b.zip outputs/baseline/*chart* outputs/sft/*chart*.jsonl results/*chart* outputs/sft/qwen3b-r16-chart/train_stats.json outputs/sft/qwen3b-r16-chart/adapter
!ls -la /kaggle/working/round4b.zip
```

### 4-A2: 차트 빈칸 채우기 — 돌리지 않는다 (규칙으로 바꿈)

4-B 결과: 차트에 나이가 없는 29건 중 16건에서 나이를 지어냈다. 처음에는 아래처럼 14B 모델이 채우게 했지만, LLM 판단 대신
규칙으로 채우기로 바꿨다. 이름·나이·성별은 이제 로컬의 `src/demographics.py`가 채우고(값마다 출처 기록), 그 결과가 이미
`data/sft_chart`, `data/chart`에 들어 있다 — **바로 4-C로 간다.** 아래는 기록용으로 남긴 예전 계획이다.
- 이름·나이·성별: 항상 채운다 (대화 "hi, sarah" → Sarah, 학생이면 20대 초반 등)
- 그 밖의 항목: 대화가 언급하는데 차트에 없을 때만 ("혈압 괜찮네요" → 혈압). 기록에 값이 있으면 그 값, 없으면 지어낸 값
- 대화가 언급하지 않는 항목은 그대로 "not recorded"

```python
# 셀 1: 코드 + vLLM
%cd /kaggle/working
!git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git || (cd healthcare && git pull -q)
%cd /kaggle/working/healthcare/clinical-note-llm
!pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 2: 14B 모델로 채우기 → 압축
%cd /kaggle/working/healthcare/clinical-note-llm
!python kaggle/09_complete_chart.py
!zip -q -r /kaggle/working/round4a2.zip data/chart/extracted/completion_*.jsonl
!ls -la /kaggle/working/round4a2.zip
```

### 4-C: 채운 차트로 다시 학습 (약 1시간 반, 새 세션)

모든 차트가 같은 8개 항목을 갖고, 이름·나이·성별은 항상 채워져 있다(규칙, `src/demographics.py`). 평가는 두 가지:
채운 차트(실제 EHR 조건)와, 채운 값을 지운 차트(`chart_text_strict`, "모르면 쓰지 않는가" 스트레스 테스트).

```python
# 셀 1: 코드 + 학습 라이브러리
%cd /kaggle/working
!git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git || (cd healthcare && git pull -q)
%cd /kaggle/working/healthcare/clinical-note-llm
!pip install -q trl peft bitsandbytes && pip uninstall -y -q torchao
```

```python
# 셀 2: 학습(약 25분) → 병합
%cd /kaggle/working/healthcare/clinical-note-llm
!torchrun --nproc_per_node 2 kaggle/02_sft.py --run-name qwen3b-r16-chart2 --data-dir data/sft_chart --max-length 5200 && python kaggle/02_sft.py --run-name qwen3b-r16-chart2 --merge-only
```

```python
# 셀 3: vLLM 설치 — 끝나면 Run → Restart & clear cell outputs
!pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 4: 테스트 120건 — 채운 차트 / 채운 값을 지운 차트
%cd /kaggle/working/healthcare/clinical-note-llm
M = "outputs/sft/qwen3b-r16-chart2/merged"
!python kaggle/01_baseline.py --model Qwen/Qwen2.5-3B-Instruct --mode json --tag Qwen2.5-3B-Instruct-chart2-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart
!python kaggle/01_baseline.py --model {M} --mode json --group sft --tag qwen3b-r16-chart2-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart
!python kaggle/01_baseline.py --model {M} --mode json --group sft --tag qwen3b-r16-chart2-strict-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart --chart-field chart_text_strict
!python src/faithfulness.py outputs/baseline/Qwen2.5-3B-Instruct-chart2-fp0.4_json.jsonl outputs/sft/qwen3b-r16-chart2-fp0.4_json.jsonl --chart-dir data/chart --out results/faithfulness_chart2.json
!python src/faithfulness.py outputs/sft/qwen3b-r16-chart2-strict-fp0.4_json.jsonl --chart-dir data/chart --chart-field chart_text_strict --out results/faithfulness_chart2_strict.json
```

```python
# 셀 5: 결과 압축
%cd /kaggle/working/healthcare/clinical-note-llm
!zip -q -r /kaggle/working/round4c.zip outputs/baseline/*chart2* outputs/sft/*chart2*.jsonl results/*chart2* outputs/sft/qwen3b-r16-chart2/train_stats.json
!ls -la /kaggle/working/round4c.zip
```

### 4-D: 빈칸 = 없음을 가르친다 (약 1시간, 새 세션)

> 결과 (2026-10-28): 채운 차트 ROUGE-1 58.0, 근거 없는 숫자 4.2%, 지어낸 나이 0. 빈칸 차트에서 지어낸 나이 15 → 11/29건.
> README "차트 3차" 참고. 차트 3차 모델을 최종 모델로 쓴다.

4-C 결과: 차트의 나이 칸이 비어 있는 29건 중 15건에서 여전히 나이를 지어냈다. 학습 차트에는 나이가 항상 있어서 빈칸을
본 적이 없기 때문이다. 이번에는 빈칸을 "not recorded" 대신 정말 비워 두고("Age:"), 학습 데이터에 짝을 23개 더했다.
같은 대화를 놓고 나이 칸을 비운 차트와 나이를 지운 정답을 함께 넣는다(대화에서 나이를 말하지 않은 사례 중 절반,
`build_chart_data.blank_age_example`). 목표: 빈칸 차트에서 지어낸 나이 15 → 0, 채운 차트 성능은 유지.

```python
# 셀 1: 코드 + 학습 라이브러리
%cd /kaggle/working
!git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git || (cd healthcare && git pull -q)
%cd /kaggle/working/healthcare/clinical-note-llm
!pip install -q trl peft bitsandbytes && pip uninstall -y -q torchao
```

```python
# 셀 2: 학습(약 30분, 학습 예시 79 → 98건) → 병합
%cd /kaggle/working/healthcare/clinical-note-llm
!torchrun --nproc_per_node 2 kaggle/02_sft.py --run-name qwen3b-r16-chart3 --data-dir data/sft_chart --max-length 5200 && python kaggle/02_sft.py --run-name qwen3b-r16-chart3 --merge-only
```

```python
# 셀 3: vLLM 설치 — 끝나면 Run → Restart & clear cell outputs
!pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 4: 테스트 120건 — 채운 차트 / 빈칸 차트
%cd /kaggle/working/healthcare/clinical-note-llm
M = "outputs/sft/qwen3b-r16-chart3/merged"
!python kaggle/01_baseline.py --model {M} --mode json --group sft --tag qwen3b-r16-chart3-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart
!python kaggle/01_baseline.py --model {M} --mode json --group sft --tag qwen3b-r16-chart3-strict-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart --chart-field chart_text_strict
!python src/faithfulness.py outputs/sft/qwen3b-r16-chart3-fp0.4_json.jsonl --chart-dir data/chart --out results/faithfulness_chart3.json
!python src/faithfulness.py outputs/sft/qwen3b-r16-chart3-strict-fp0.4_json.jsonl --chart-dir data/chart --chart-field chart_text_strict --out results/faithfulness_chart3_strict.json
```

```python
# 셀 5: 결과 압축
%cd /kaggle/working/healthcare/clinical-note-llm
!zip -q -r /kaggle/working/round4d.zip outputs/sft/*chart3*.jsonl results/*chart3* outputs/sft/qwen3b-r16-chart3/train_stats.json
!ls -la /kaggle/working/round4d.zip
```

### 4-E: 당일 검사를 뺀 차트로 다시 학습 (약 1시간, 새 세션)

차트를 정답 기록에서 거꾸로 만들다 보니, 이번 진료 중에 찍은 검사("X-rays taken in office today reveal...")가 진료 전
차트에 들어가 있었다. 모델이 써야 할 RESULTS 구역을 입력으로 미리 보여 준 셈이라 차트 3차의 ROUGE 58.0·O-results 65가
부풀었을 수 있다. `build_chart_data.same_day`로 이런 항목 23개(학습 12, 검증 3, 테스트 8)를 빼고 다시 만든 데이터로
같은 설정으로 학습한다. 나머지(빈칸 짝 19건 등)는 차트 3차와 같다.

같은 세션에서 **영어 중심 모델 Llama-3.2-3B-Instruct**도 같은 데이터·설정으로 학습해 비교한다. Qwen2.5는 중국어·영어
중심의 다국어 모델이고, Llama 3.2는 영어 중심이다. 크기(3B)가 같아 모델 계열 차이만 본다. 전체 약 2.5시간.

준비: Hugging Face에서 meta-llama/Llama-3.2-3B-Instruct 페이지의 라이선스 동의, Kaggle Secrets에 `HF_TOKEN`.

```python
# 셀 1: 코드 + 학습 라이브러리 + HF 로그인(Llama는 동의한 계정만 받을 수 있다)
%cd /kaggle/working
!git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git || (cd healthcare && git pull -q)
%cd /kaggle/working/healthcare/clinical-note-llm
!pip install -q trl peft bitsandbytes && pip uninstall -y -q torchao
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

```python
# 셀 2: Qwen 학습(약 30분) → 병합
%cd /kaggle/working/healthcare/clinical-note-llm
!torchrun --nproc_per_node 2 kaggle/02_sft.py --run-name qwen3b-r16-chart4 --data-dir data/sft_chart --max-length 5200 && python kaggle/02_sft.py --run-name qwen3b-r16-chart4 --merge-only
```

```python
# 셀 2-L: Llama 학습(약 30분) → 병합
%cd /kaggle/working/healthcare/clinical-note-llm
L = "meta-llama/Llama-3.2-3B-Instruct"
!torchrun --nproc_per_node 2 kaggle/02_sft.py --model {L} --run-name llama3b-r16-chart4 --data-dir data/sft_chart --max-length 5200 && python kaggle/02_sft.py --model {L} --run-name llama3b-r16-chart4 --merge-only
```

```python
# 셀 3: vLLM 설치 — 끝나면 Run → Restart & clear cell outputs
!pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 4: 테스트 120건 — 채운 차트 / 빈칸 차트
%cd /kaggle/working/healthcare/clinical-note-llm
M = "outputs/sft/qwen3b-r16-chart4/merged"
!python kaggle/01_baseline.py --model {M} --mode json --group sft --tag qwen3b-r16-chart4-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart
!python kaggle/01_baseline.py --model {M} --mode json --group sft --tag qwen3b-r16-chart4-strict-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart --chart-field chart_text_strict
!python src/faithfulness.py outputs/sft/qwen3b-r16-chart4-fp0.4_json.jsonl --chart-dir data/chart --out results/faithfulness_chart4.json
!python src/faithfulness.py outputs/sft/qwen3b-r16-chart4-strict-fp0.4_json.jsonl --chart-dir data/chart --chart-field chart_text_strict --out results/faithfulness_chart4_strict.json
```

```python
# 셀 4-L: Llama — 학습 전 / 학습 후(채운 차트) / 학습 후(빈칸 차트). 셀 3 재시작 뒤라 HF_TOKEN을 다시 넣는다
%cd /kaggle/working/healthcare/clinical-note-llm
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
L, M = "meta-llama/Llama-3.2-3B-Instruct", "outputs/sft/llama3b-r16-chart4/merged"
!python kaggle/01_baseline.py --model {L} --mode json --group baseline --tag Llama-3.2-3B-Instruct-chart4-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart
!python kaggle/01_baseline.py --model {M} --mode json --group sft --tag llama3b-r16-chart4-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart
!python kaggle/01_baseline.py --model {M} --mode json --group sft --tag llama3b-r16-chart4-strict-fp0.4 --frequency-penalty 0.4 --chart-dir data/chart --chart-field chart_text_strict
!python src/faithfulness.py outputs/baseline/Llama-3.2-3B-Instruct-chart4-fp0.4_json.jsonl outputs/sft/llama3b-r16-chart4-fp0.4_json.jsonl --chart-dir data/chart --out results/faithfulness_llama_chart4.json
!python src/faithfulness.py outputs/sft/llama3b-r16-chart4-strict-fp0.4_json.jsonl --chart-dir data/chart --chart-field chart_text_strict --out results/faithfulness_llama_chart4_strict.json
```

```python
# 셀 5: 결과 압축
%cd /kaggle/working/healthcare/clinical-note-llm
!zip -q -r /kaggle/working/round4e.zip outputs/sft/*chart4*.jsonl outputs/baseline/*chart4*.jsonl results/*chart4* outputs/sft/*-chart4/train_stats.json
!ls -la /kaggle/working/round4e.zip
```

막히면
- 셀 2-L에서 401/403(gated repo): HF에서 Llama 3.2 라이선스 동의가 안 됐거나 `HF_TOKEN`이 비었다.
- 셀 2-L에서 GPU 메모리 부족: Llama 어휘가 커서(128k) Qwen보다 메모리를 더 쓴다. `--max-length 4096`으로 다시 실행한다.

## 3차: DPO — "대화와 반대로 쓰기"·지어낸 사실 줄이기

흐름: 2차 SFT 모델로 학습 대화마다 후보 8개 생성(vLLM) → 14B 판정 모델이 후보마다 "대화와 반대"·"대화에 없는 사실"을
찾음 → 좋은 답·나쁜 답 짝 만들기(CPU) → DPO 학습 → 테스트 평가. 같은 판정 모델로 학습 전·1차·2차 SFT 테스트 결과도
채점해 "사실과 반대" 건수를 처음으로 120건 전체에서 비교한다. 전체 약 2~3시간.

```python
# 셀 1: 코드 + 학습 라이브러리
%cd /kaggle/working
!git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git || (cd healthcare && git pull -q)
%cd /kaggle/working/healthcare/clinical-note-llm
!pip install -q trl peft bitsandbytes && pip uninstall -y -q torchao
```

```python
# 셀 2: 2차 SFT 모델이 없으면(새 세션) 다시 학습·병합 — 약 25분. 있으면 바로 넘어간다
%cd /kaggle/working/healthcare/clinical-note-llm
import os
if not os.path.exists("outputs/sft/qwen3b-r16-grounded/merged/config.json"):
    !torchrun --nproc_per_node 2 kaggle/02_sft.py --run-name qwen3b-r16-grounded --data-dir data/sft_grounded && python kaggle/02_sft.py --run-name qwen3b-r16-grounded --merge-only
!ls outputs/sft/qwen3b-r16-grounded/merged
```

```python
# 셀 3: vLLM 설치 — 끝나면 Run → Restart & clear cell outputs
!pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 4: 후보 생성(약 10분) → 판정(14B, GPU 2장, 약 30~40분) → 짝 만들기
%cd /kaggle/working/healthcare/clinical-note-llm
!python kaggle/05_dpo_candidates.py --model outputs/sft/qwen3b-r16-grounded/merged
!python kaggle/06_judge.py outputs/dpo/candidates_train.jsonl outputs/dpo/candidates_valid.jsonl outputs/baseline/Qwen2.5-3B-Instruct-fp0.4_json.jsonl outputs/sft/qwen3b-r16-fp0.4_json.jsonl outputs/sft/qwen3b-r16-grounded-fp0.4_json.jsonl
!python src/build_dpo_pairs.py
```

```python
# 셀 5: DPO 학습(약 30~60분) → 병합
%cd /kaggle/working/healthcare/clinical-note-llm
!torchrun --nproc_per_node 2 kaggle/07_dpo.py && python kaggle/07_dpo.py --merge-only
```

```python
# 셀 6: 테스트 120건 평가 + 근거 검사 + 판정
%cd /kaggle/working/healthcare/clinical-note-llm
!python kaggle/01_baseline.py --model outputs/dpo/dpo-b0.1/merged --mode json --group dpo --tag dpo-b0.1-fp0.4 --frequency-penalty 0.4
!python src/faithfulness.py outputs/baseline/Qwen2.5-3B-Instruct-fp0.4_json.jsonl outputs/sft/qwen3b-r16-fp0.4_json.jsonl outputs/sft/qwen3b-r16-grounded-fp0.4_json.jsonl outputs/dpo/dpo-b0.1-fp0.4_json.jsonl
!python kaggle/06_judge.py outputs/dpo/dpo-b0.1-fp0.4_json.jsonl
```

```python
# 셀 7: 결과 압축(어댑터 포함, 약 250MB) → Output 패널에서 받아 clinical-note-llm/ 에 푼다
%cd /kaggle/working/healthcare/clinical-note-llm
!zip -q -r /kaggle/working/round3.zip data/dpo outputs/dpo outputs/judge results/judge_* results/dpo_* results/faithfulness.json outputs/sft/qwen3b-r16-grounded/adapter -x "*/merged/*"
!ls -la /kaggle/working/round3.zip
```

막히면
- 셀 5에서 trl 관련 오류(vLLM 설치가 transformers 버전을 바꿔서): 오류를 붙여 주세요. 대개 버전 하나만 맞추면 된다.
- 셀 5에서 GPU 메모리 부족: `--load-4bit`을 붙여 다시 실행한다.

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
