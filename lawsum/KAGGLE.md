# Kaggle에서 모델 비교 (GPU T4 x2, Internet On)

같은 프롬프트(12차)와 같은 후처리로 모델만 바꿔 62건을 요약한다. 로컬 기록(CPU, Llama-3.1-8B 4bit GGUF)과
실행 방식이 달라지므로 Llama-3.1-8B도 같은 방식(GPU, FP16)으로 다시 돌려 세 모델을 같은 조건에서 비교한다.

| 태그 | 모델 | 비고 |
|---|---|---|
| llama-3.1-8b | meta-llama/Llama-3.1-8B-Instruct | 기존 모델. Hugging Face에서 라이선스 동의 필요 |
| kanana-1.5-8b | kakaocorp/kanana-1.5-8b-instruct-2505 | 카카오, 한국어 특화, Apache 2.0, Llama 구조 |
| ax-4.0-light | skt/A.X-4.0-Light | SK텔레콤, 한국어 특화, Apache 2.0, Qwen2.5 구조(7B) |

준비: Hugging Face에서 https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 라이선스 동의 → Kaggle 노트북
**Add-ons → Secrets**에 `HF_TOKEN` 등록(읽기 토큰).

```python
# 셀 1: 코드 + vLLM — 끝나면 Run → Restart & clear cell outputs
%cd /kaggle/working
!git clone -q -b lawsum https://github.com/ejrtn/healthcare.git healthcare-lawsum || (cd healthcare-lawsum && git pull -q)
!pip install -q vllm && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 2: 모델마다 약 10~15분 (다운로드 포함). 모델마다 별도 프로세스로 띄워 GPU 메모리를 확실히 돌려받는다
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
%cd /kaggle/working/healthcare-lawsum/lawsum
!python kaggle_summarize.py --model kakaocorp/kanana-1.5-8b-instruct-2505 --tag kanana-1.5-8b
!python kaggle_summarize.py --model skt/A.X-4.0-Light --tag ax-4.0-light
!python kaggle_summarize.py --model meta-llama/Llama-3.1-8B-Instruct --tag llama-3.1-8b
```

```python
# 셀 3: 결과 압축 → Output 패널에서 받아 lawsum/ 에 푼다
%cd /kaggle/working/healthcare-lawsum/lawsum
!zip -q /kaggle/working/lawsum_models.zip data/summarized_*.json data/run_*.json
!ls -la /kaggle/working/lawsum_models.zip
```

로컬에서 채점 (CPU, 임베딩 유사도):

```bash
python evaluate.py --summarized data/summarized_kanana-1.5-8b.json --out data/evaluated_kanana-1.5-8b.json
```

## 2차: A.X 숫자 오류 원인 가르기 (약 10분)

A.X 요약 41/62건에서 날짜·사건번호의 한 자리가 빠졌다(`check_citations.py`). 샘플링(temperature 0.2) 때문인지
확인하려고 temperature 0으로 다시 돌린다. Llama도 0으로 돌려 같은 조건의 기준을 만든다.

```python
# 셀 1(설치)을 다시 실행한 뒤
%cd /kaggle/working/healthcare-lawsum/lawsum
!git pull -q
!python kaggle_summarize.py --model skt/A.X-4.0-Light --tag ax-4.0-light-t0 --temperature 0
!python kaggle_summarize.py --model meta-llama/Llama-3.1-8B-Instruct --tag llama-3.1-8b-t0 --temperature 0
!zip -q /kaggle/working/lawsum_t0.zip data/summarized_*-t0.json data/run_*-t0.json
```

로컬 채점: `python check_citations.py data/summarized_ax-4.0-light-t0.json` + `python evaluate.py ...`

## 3차: 숫자 누락 원인 — vLLM 출력 변환 vs FP16 정밀도 (약 30분)

temperature 0에서도 같은 자리가 빠졌다(43/62). 틀린 표기가 나오는 3건만 앞부분까지 생성해 비교한다.

```python
# 셀 1(설치) 후
%cd /kaggle/working/healthcare-lawsum/lawsum
!git pull -q
!python diagnose_digits.py --mode vllm          # vLLM 출력 vs 같은 토큰을 HF 토크나이저로 다시 푼 글자
!python diagnose_digits.py --mode hf-gpu-fp16   # vLLM 없이 transformers, GPU FP16
!python diagnose_digits.py --mode hf-cpu-bf16   # transformers, CPU BF16 (학습 때와 같은 정밀도, 건당 수 분)
!zip -q /kaggle/working/lawsum_diag.zip data/diagnose_*.json
```

읽는 법: vllm에서 `decode_mismatch`가 true면 출력 변환 문제. hf-gpu-fp16도 틀리고 hf-cpu-bf16은 맞으면 FP16 정밀도 문제.
셋 다 틀리면 모델 자체의 버릇.

## 4차: 처음 보는 판례 99건으로 최종 평가 (약 15분, GPU)

프롬프트·후처리 규칙은 모두 `cases.json` 62건 점수를 보며 정했다. 그 점수가 62건에 맞춘 값일 수 있어서, 같은 조건으로
새로 모은 판례(`data/heldout_cases.json`, 10개 분야, 기존과 겹침 0건)로 **아무것도 바꾸지 않고 한 번만** 평가한다.

```python
# 셀 1: 코드 + vLLM — 끝나면 Run → Restart & clear cell outputs
%cd /kaggle/working
!git clone -q -b lawsum https://github.com/ejrtn/healthcare.git healthcare-lawsum || (cd healthcare-lawsum && git pull -q)
!pip install -q vllm && pip uninstall -y -q torchaudio torchao
```

```python
# 셀 2: A.X(최종 설정)와 Llama(비교) — temperature 0
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
%cd /kaggle/working/healthcare-lawsum/lawsum
!git pull -q
!python kaggle_summarize.py --cases data/heldout_cases.json --model skt/A.X-4.0-Light --tag heldout-ax-4.0-light-t0 --temperature 0
!python kaggle_summarize.py --cases data/heldout_cases.json --model meta-llama/Llama-3.1-8B-Instruct --tag heldout-llama-3.1-8b-t0 --temperature 0
!zip -q /kaggle/working/lawsum_heldout.zip data/summarized_heldout-*.json data/run_heldout-*.json
!ls -la /kaggle/working/lawsum_heldout.zip
```

로컬 채점: 원문 대조 보정(`check_citations.py --fix`) → `evaluate.py --baseline data/heldout_baseline.json`
