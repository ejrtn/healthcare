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
