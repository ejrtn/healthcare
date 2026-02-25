# 의료 영상 AI: 흉부 X-ray 및 복부 CT 판독 프로젝트

복부 외상 탐지(CT) 및 흉부 질환 분류(X-ray)를 위한 딥러닝 프로젝트입니다. 고급 전처리부터 정교한 모델 아키텍처 설계까지, 의료용 AI 개발의 전 과정을 포함하고 있습니다.

## 🚀 주요 기술적 특징

### 1. 2.5D 복부 외상 탐지 (CT)
- **아키텍처**: ConvNeXt-Tiny Backbone + Transformer Encoder + Gated Heads.
- **기술적 혁신**:
    - **문맥 분석 (Contextual Analysis)**: Transformer Encoder를 활용하여 64장의 연속된 CT 슬라이스 간의 공간적 관계를 포착합니다.
    - **게이트 멀티헤드 전략 (Gated Multi-Head Strategy)**: 전체 부상 여부를 판단하는 "Suspicion Head"가 개별 장기 분류기를 제어(Gating)합니다. 이를 통해 전체 부상 확률에 따라 장기 손상 판독을 조절함으로써 오탐율(False Positive)을 획기적으로 낮췄습니다.
    - **최적화**: 레이어별 학습률 감소(LLRD) 전략을 사용하여 깊은 비전 백본 모델을 안정적으로 미세 조정(Fine-tuning)했습니다.
- **기술 스택**: PyTorch, MONAI, TIMM, TorchMetrics.

### 2. 멀티 데이터셋 벤치마킹 (X-ray)
- **모델**: DenseNet-121 기반 커스텀 멀티라벨 헤드.
- **연구 내용**: 두 개의 주요 데이터셋(**NIH Chest X-ray** 및 **CheXpert**)을 표준화된 라벨과 균형 잡힌 로그 스케일 가중치를 사용하여 성능을 비교 분석했습니다.
- **최적화**: 전략적 레이어 동결 해제(Unfreezing) 및 스케줄링된 학습률 감소를 통한 전이 학습 최적화를 수행했습니다.

## 📂 저장소 구조

```text
├── src/
│   ├── data/       # MONAI 기반 데이터셋 및 의료용 영상 전처리
│   ├── models/     # CT (ConvNeXt+Transformer) 및 X-ray (DenseNet) 아키텍처
│   ├── engine/     # LLRD 및 Gradient Accumulation이 적용된 트레이너
│   └── utils/      # 시각화 및 의료 영상 평가 지표
├── experiments/    # 실험 과정이 담긴 Jupyter Notebook 아카이브
├── assets/         # 모델 가중치, 학습 히스토리 및 시각화 결과물
├── README.md       # 프로젝트 소개 (현재 파일)
└── requirements.txt # 환경 설정 및 재현 가이드
```

## 📊 성능 시각화

### CT 학습 히스토리
![CT History](assets/ct_result_history_6_4.png)
*그림: ConvNeXt + Transformer 모델의 최고 성능(Mean AUC 0.752)을 기록한 학습 과정.*

### X-ray 데이터셋 비교
![X-ray Results](assets/x-ray.png)
*그림: NIH와 CheXpert 데이터셋 간의 주요 흉부 질환 판독 성능 비교 분석.*

## 🛠️ 설치 및 사용법

1. **환경 설정**:
   ```bash
   pip install -r requirements.txt
   ```

2. **모듈 활용**:
   - 모델 로드: `from src.models.model_ct import CTConvNeXtModel`
   - 데이터 로드: `from src.data.dataset import CTDataset, get_ct_transforms`
   - 학습 엔진: `from src.engine.trainer import CTTrainer`

## 👨‍💻 작성자
**의료 AI 엔지니어 지망생**
- 의료 영상 기반 딥러닝 연구
- volumetric 데이터 처리를 위한 Transformer 구조 전문화
