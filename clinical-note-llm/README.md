# Clinical Note LLM — 진료 대화 → 구조화된 진료기록

> 진행 중 (2026-10-27 시작, 목표 완료 2026-11-10)

의사-환자 진료 대화 전체(평균 약 1,200단어)를 읽고 SOAP 형식의 진료기록을
JSON으로 생성하는 소형 LLM을 직접 학습(SFT)하고, vLLM으로 서빙 최적화까지
측정하는 프로젝트.

## 데이터

| 데이터셋 | 용도 | 규모 | 라이선스 |
|---|---|---|---|
| [ACI-BENCH](https://github.com/wyim/aci-bench) | 학습 + **최종 평가** (진료 1회 전체 대화 → 전체 기록) | train 67 / valid 20 / test 120 | CC BY 4.0 |
| [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) | 학습 보조 (짧은 대화 → 기록 섹션 하나) | train 1,201 / valid 100 / test 400 | CC BY 4.0 |

`bash scripts/download_data.sh`로 받는다 (커밋 고정).

데이터 분석 메모:
- ACI-BENCH 대화 길이: 중앙값 약 1,240단어, 최대 3,050단어 / 진료기록 중앙값 약 400단어
- ACI-BENCH 기록의 섹션 제목이 사건마다 다르다 (CHIEF COMPLAINT, HPI, ROS,
  PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN 등) → SOAP 3개 필드로 정규화 필요
- ACI-BENCH에는 같은 대화의 **ASR 원본 / ASR 교정 / 사람 전사** 버전이 있다
  → 음성인식 오류에 대한 강건성 측정 가능
- ACI-BENCH 저장소에 공식 평가 코드(ROUGE, BERTScore, UMLS 기반 MEDCON)와
  논문 기존 모델(BART, LED 등) 예측 결과가 있다 → 같은 기준으로 비교

## 평가

- 출력 JSON 스키마 = ACI-BENCH 공식 평가가 나누는 4구역
  (`subjective`, `objective_exam`, `objective_results`, `assessment_and_plan`).
  구역 분리는 공식 코드의 SectionTagger를 그대로 옮겨 쓴다 (`src/sectiontagger.py`).
  정답 기록을 4구역으로 나눴다가 다시 합쳐도 67/67건 동일, 원문 보존율 99.8%.
- 채점기(`src/evaluate.py`): 전체 기록 + 구역별 ROUGE. 공식 결과 파일과 비교하면
  ROUGE-1이 0.2점 정도 어긋난다(BioBART test2: 39.00 vs 공식 39.20). 그래서 논문 표를
  인용하지 않고, 저장소에 있는 기존 모델 예측을 **같은 채점기로 재채점**해 비교한다.

### 기존 모델 재채점 (test1~3, 120건) — `src/score_baselines.py`

| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | S | O-exam | O-results | A&P |
|---|---|---|---|---|---|---|---|
| BART+SAMSum FT (division) | 52.77 | 24.61 | 28.87 | 48.87 | 41.74 | 22.33 | 41.29 |
| GPT-4 (zero-shot) | 51.10 | 21.97 | 29.55 | 42.94 | 47.32 | 32.13 | 40.10 |
| ChatGPT (zero-shot) | 45.27 | 18.31 | 25.76 | 31.08 | 41.90 | 28.44 | 33.80 |
| BART+SAMSum FT | 40.20 | 18.74 | 22.70 | 46.02 | 4.05 | 27.15 | 1.28 |
| BioBART FT | 38.80 | 17.69 | 21.75 | 45.95 | 3.23 | 25.87 | 0.46 |
| Transcript copy | 32.65 | 12.79 | 19.10 | 17.13 | 1.71 | 26.67 | 0.00 |
| LED FT | 28.91 | 5.94 | 10.81 | 24.07 | 1.71 | 26.67 | 0.00 |

구역 열은 구역별 ROUGE-1. 목표: 소형 오픈 모델 + SFT로 GPT-4(51.1) 수준 도달.

## 계획

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | Kaggle T4에서 vLLM 동작 확인 (`kaggle/00_vllm_smoke_test.py`) | ⏳ |
| 1 | 출력 스키마·평가 파이프라인 ✅ / 프롬프트 기준선 측정 | 진행 중 |
| 2 | QLoRA SFT | |
| 3 | vLLM 서빙 벤치마크 (FP16 vs 4bit, 동시 요청, Tensor Parallel, structured output) | |
| 4 | FastAPI 비동기 API + Docker + 부하 테스트 | |
| 5 | (여유 시) DPO로 환각 감소 / 오류 분석 | |
