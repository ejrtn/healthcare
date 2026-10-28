# 데이터 구조

원본은 읽기만 하고, 전처리 결과는 모두 스크립트로 다시 만들 수 있게 단계별로 따로 둔다.
어떤 단계든 지우고 스크립트를 다시 돌리면 같은 결과가 나온다(LLM 출력은 temperature 0).

```
data/
├── raw/                    ① 원본 — 수정 금지, 저장소에 올리지 않음
│   ├── aci-bench/            github.com/wyim/aci-bench @ b909b2b (CC BY 4.0)
│   └── MTS-Dialog/           github.com/abachaa/MTS-Dialog @ 3ff0801 (CC BY 4.0, 현재 미사용)
│
├── processed/              ② 원본을 같은 형식(jsonl)으로만 바꾼 것 — 내용 변경 없음
│
├── chart/extracted/        ③ LLM 출력 — 고치지 않고 보관 (Kaggle GPU로 만들어 다시 만들기 비싸다)
│   ├── chart_{split}.jsonl         정답 기록에서 복원한 진료 전 차트
│   └── completion_{split}.jsonl    빈 이름·나이·성별 + 대화가 가리키는데 빠진 항목을 채운 값 (기록에 있으면 그 값, 없으면 지어냄)
│
└── 전처리 결과 ④ — 언제든 지우고 다시 만들 수 있다
    ├── sft/                  학습 데이터: 대화 → 정답 기록 (원본 정답 그대로)
    ├── sft_grounded/         정답에서 대화에 없는 나이·성을 지운 사본 (원본·sft는 그대로)
    ├── chart/{split}.jsonl   ③을 걸러내고 정리한 차트 (+ 채우기 전 chart_text_strict)
    ├── sft_chart/            학습 데이터: 차트 + 대화 → 정답 기록 (원본 정답 그대로)
    └── dpo/                  DPO 좋은 답·나쁜 답 짝
```

| 단계 | 만드는 스크립트 | 입력 |
|---|---|---|
| ① raw | `scripts/download_data.sh` (커밋 고정) | 인터넷 |
| ② processed | `src/prepare_data.py` | ① |
| ③ chart/extracted/chart_* | `kaggle/08_extract_chart.py` (GPU) | ②, ④ sft |
| ③ chart/extracted/completion_* | `kaggle/09_complete_chart.py` (GPU) | ②, ④ chart |
| ④ sft | `src/build_sft_data.py` | ②, ① train_virtscribe_asr.csv |
| ④ sft_grounded | `src/build_sft_data.py --grounded` | ② |
| ④ chart, sft_chart | `src/build_chart_data.py` | ②, ③, ④ sft |
| ④ dpo | `src/build_dpo_pairs.py` | ④ sft_grounded, outputs/dpo, outputs/judge |

원칙
- 정답 기록을 바꾸는 전처리(sft_grounded)는 항상 **새 폴더에 사본**으로 만든다. 원본과 다른 전처리 결과는 그대로 남는다.
- LLM 출력(③)에서 고칠 것이 보여도 ③을 직접 고치지 않는다. 걸러내기 규칙을 `src/build_chart_data.py`에 추가하고 ④를 다시 만든다
  (예: "[value]" 같은 자리 표시 115개 제거, 성별 표기 통일).
- 원본이 그대로인지 확인: `bash scripts/verify_raw.sh`
