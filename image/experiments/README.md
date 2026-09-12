# ct-convnext-tiny-s224_19

## 실험 목적

**v11(baseline)에서 옵티마이저만 교체한 비교 실험.**

v11은 `LAYER_DECAY`를 선언만 하고 실제로는 `AdamW(model.parameters(), ...)`로 전체 파라미터를 같은 LR로 묶어버려 미적용이었다.  
v19는 `timm.optim.create_optimizer_v2(..., layer_decay=LAYER_DECAY)`로 실제 적용해서, epoch 2에 backbone을 통째로 동결 해제할 때 얕은 층은 거의 안 흔들리고 깊은 층 위주로만 CT 도메인에 적응하도록 바꿨다.

나머지 설정(데이터, augmentation, loss, batch size, epoch 수)은 v11과 전부 동일하게 유지해서 **이 변경 하나의 효과만 비교**할 수 있게 했다.

---

## 태스크

**any_injury 이진 분류** — 복부 CT에서 장기 손상이 있는지(1) 없는지(0)만 판단.

- 입력: 전처리된 CT 시리즈 (64장 슬라이스, 각 224x224, 3채널)
- 출력: [정상 확률, 이상 확률] 2-class softmax

---

## 주요 경로 변수

| 변수 | 역할 |
|---|---|
| `BASE_DIR` | 원본 데이터 기본 경로 (CSV, parquet 읽기용) |
| `SAVE_DIR` | 이미 완료된 전처리 npy 파일 경로 (새로 생성하지 않음) |
| `MONAI_MODEL_SAVE_PATH` | 학습된 모델 .pth 저장 경로 (에포크별 _ep{N}.pth 생성) |
| `MONAI_MODEL_SAVE_PATH_CONTINUE` | 이어서 학습할 체크포인트 경로 (빈 문자열이면 처음부터 시작) |

---

## 셀 구조

| # | 셀 이름 | 내용 |
|---|---|---|
| 1 | pip install | monai 설치 |
| 2 | 임포트 | 라이브러리 import |
| 3 | 설정 & 경로 | BASE_DIR, SAVE_DIR, MONAI_MODEL_SAVE_PATH, MONAI_MODEL_SAVE_PATH_CONTINUE, 하이퍼파라미터 |
| 4 | 데이터 준비 | CSV/parquet 읽기 → data_dicts 구성 (any_injury만) → train/val split |
| 5 | 모델 & 파이프라인 | Timm_Model, LoadNpyTransformd, monai_train_pipeline, monai_val_pipeline |
| 6 | 학습 함수 | process_one_item, evaluate, train |
| 7 | 실행 | 전처리 경로 매핑 → 파이프라인 생성 → train() 호출 |

---

## 모델 아키텍처

입력 (B, 64, 3, 224, 224)
  → chunk_size=8 단위로 슬라이스 처리
  → ConvNeXt-Tiny backbone (pretrained, num_classes=0)
  → 슬라이스별 768-dim 특징 벡터
  → Attention Pooling (이상 징후 슬라이스에 높은 가중치)
  → (B, 768) 통합 벡터
  → suspicion_head: Linear(768→256) → LayerNorm → ReLU → Dropout(0.1) → Linear(256→2)
  → {'any_injury': (B, 2)}

### 학습 2단계 전략

| Epoch | 상태 |
|---|---|
| 0~1 | Backbone 동결, head만 학습 |
| 2+ | Backbone 동결 해제, layer_decay LR 그대로 유지 (v11처럼 일괄 덮어쓰기 없음) |

---

## 학습 설정

| 항목 | 값 |
|---|---|
| Backbone | convnext_tiny (pretrained) |
| 입력 크기 | (64, 224, 224) 슬라이스 x 3채널 |
| Optimizer | create_optimizer_v2 (AdamW, layer_decay=0.8) |
| LR | 1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=3) |
| Loss | CrossEntropyLoss (weight=[1.0, 5.0], label_smoothing=0.05) |
| Batch size | 2 |
| Epochs | 25 |
| GPU | NVIDIA Tesla T4 x2 (DataParallel) |
| AMP | GradScaler + autocast |

Loss 선택 이유: [정상, 이상] 2-class Softmax 구조이므로 CrossEntropyLoss 사용.
클래스 불균형 보정을 위해 이상 클래스 가중치를 5배로 설정.

---

## 저장 파일

| 파일 | 내용 |
|---|---|
| monai_ct_convnext_v19_ep{N}.pth | 에포크별 전체 체크포인트 (model, optimizer, scheduler, scaler, history) |
| monai_ct_convnext_v19.pkl | 학습 히스토리 (train_loss, val_loss, auc_avg, auc_details) |

---

## 데이터 전처리 (SAVE_DIR)

전처리는 이미 완료된 상태이며 SAVE_DIR에 .npy 파일로 저장되어 있다.
process_one_item()이 원본 시리즈 경로를 SAVE_DIR/{series_id}.npy 경로로 교체한다.
파일이 없는 시리즈는 학습 목록에서 제외된다.

npy 파일 형식: (64, 224, 224, 3) 또는 (3, 64, 224, 224) - 로드 시 자동 변환.

---

## 관련 버전 비교

| 버전 | 변경 내용 |
|---|---|
| v11 | baseline (LAYER_DECAY 선언만, 실제 미적용) |
| v19 | create_optimizer_v2로 LAYER_DECAY 실제 적용 |
