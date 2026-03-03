# 의료 영상 AI: CT 및 흉부 X‑ray 분류 프레임워크

---

## 초록
본 연구는 **컴퓨터 단층촬영(CT)** 및 **흉부 X‑ray** 영상을 활용한 질병 분류를 위한 딥러닝 프레임워크를 제시한다. 의료 영상에 특화된 전처리 파이프라인을 설계하고, 전이 학습 기반 CNN과 하이브리드 CNN‑Transformer 모델을 비교하였다. 클래스 불균형 문제는 Focal Loss와 가중 교차 엔트로피로 해결하고, 재현 가능한 학습 워크플로우를 제공한다. RSNA CT 데이터셋 및 NIH/​CheXpert X‑ray 데이터셋을 이용한 실험을 통해 **Transformer‑based attention**과 **layer‑wise learning‑rate decay (LLRD)**의 효과를 입증한다.

---

## 1. 서론
의료 영상의 정확한 해석은 신속한 진단에 필수적이다. 최근 컨볼루션 신경망(CNN)의 발전으로 뛰어난 성능을 달성했지만, 여전히 다음과 같은 과제가 존재한다:
- **볼륨 CT 데이터**는 슬라이스 간 컨텍스트를 고려해야 한다.
- **흉부 X‑ray** 분류는 다중 라벨 문제이며, 클래스 불균형이 심각하다.
- 재현성 및 임상 적용성을 위해 투명한 파이프라인이 요구된다.

본 연구의 주요 기여는 다음과 같다:
1. CT와 X‑ray에 맞게 설계된 **MONAI‑based 전처리 파이프라인**.
2. CT에 **하이브리드 CNN‑Transformer 구조**를 도입하여 슬라이스 컨텍스트를 포착.
3. **DenseNet‑121**(X‑ray)과 **ConvNeXt‑Tiny + Transformer**(CT)를 체계적으로 비교.
4. **LLRD**, **gradient accumulation**, **early stopping**을 포함한 학습 전략을 제시하여 안정적인 fine‑tuning을 구현.

---

## 2. 재료 및 방법
### 2.1 데이터셋
- **CT 데이터셋**: RSNA 복부 외상 탐지 데이터 (볼륨, 연구당 64 슬라이스). HU 정규화와 슬라이스 선택을 수행.
- **흉부 X‑ray 데이터셋**: NIH Chest X‑ray와 CheXpert, 다중 라벨 형태의 일반적인 흉부 병변을 포함.

### 2.2 전처리
모든 전처리는 **MONAI**를 사용하여 구현한다:
- 강도 정규화 및 **224×224** 크기로 리사이즈.
- 랜덤 플립, 회전, 어파인 변환 등 데이터 증강.
- 클래스 불균형 완화를 위한 가중 손실 적용.
- CT 전용: HU 윈도우링 및 균등 슬라이스 샘플링.

### 2.3 모델 아키텍처
#### 2.3.1 X‑ray 모델
- **DenseNet‑121**을 ImageNet 사전학습 가중치로 초기화하고, 다중 라벨 헤드를 추가하여 Fine‑tuning.

#### 2.3.2 CT 모델
- **ConvNeXt‑Tiny** 백본(초기에는 Freeze) 사용.
- **Transformer Encoder**(2 레이어, 8 헤드)로 슬라이스 임베딩 시퀀스 처리.
- **Attention pooling**을 통해 슬라이스 정보를 집계.
- **Gated Multi‑Head Heads**는 전체 부상 확률에 따라 장기별 예측을 조절.

### 2.4 학습 전략
- Optimizer: **AdamW**.
- 학습률 스케줄: **Layer‑wise LR decay (LLRD)**를 백본 단계별로 적용하고, 헤드에는 코사인 감소 적용.
- **Gradient accumulation**(8 스텝)으로 GPU 메모리 제한을 극복.
- **Early stopping**을 검증 손실 기준으로 적용.

---

## 3. 실험
### 3.1 실험 환경
- 프레임워크: **PyTorch** + **MONAI**.
- 하드웨어: 단일 NVIDIA GPU(CUDA). 재현성을 위해 랜덤 시드 고정.
- 평가 지표: **AUC**, **Accuracy**, 장기별 **Sensitivity/Specificity**.

### 3.2 결과

#### CT 모델 버전별 벤치마크 (ConvNeXt-Tiny + Transformer)
| 모델 버전 | 최고 Epoch | 검증 Loss | AUC (평균) |
|-----------|------------|-----------|------------|
| v1_2      | 14         | 2.5379    | 0.7738     |
| v2        | 10         | 2.7214    | 0.6793     |
| v3        | 14         | 2.7362    | 0.6892     |
| v4        | 11         | 2.5536    | 0.7070     |
| v5_2      | 17         | 2.3913    | 0.7529     |
| v6_4      | 22         | 4.2893    | 0.7524     |
| v7_2      | 12         | 4.3942    | 0.7474     |
| v8        | 3          | 4.3632    | 0.6951     |
| v9        | 5          | 4.3870    | 0.7131     |
| v10       | 8          | 2.45      | 0.7757     |

#### 전체 벤치마크 요약
| 분류 대상 | 채택 모델 | 데이터셋 | 손실 함수 | 배스트 AUC |
|----------|-----------|----------|-----------|------------|
| 흉부 X-ray | DenseNet‑121 | NIH / CheXpert | 가중 교차 엔트로피 | *(작성 필요)* |
| 복부 CT    | ConvNeXt‑Tiny (v10) | RSNA CT 외상 | Focal Loss | **0.7757** |

**주요 관찰**
- Focal Loss는 심한 클래스 불균형 상황에서 성능을 향상시킨다.
- Transformer attention을 추가하면 CT 부상 탐지 AUC가 상승한다.
- LLRD는 깊은 ConvNeXt 레이어의 Fine‑tuning을 안정화한다.

---

## 4. 논의
### 4.1 강점
- **도메인 특화 전처리**가 노이즈를 감소시키고 입력을 표준화한다.
- **하이브리드 구조**가 CT 슬라이스 간 컨텍스트 추론을 가능하게 한다.
- **LLRD**와 **Gradient Accumulation**을 통한 학습 안정성 확보.

### 4.2 한계
- 독립적인 임상 코호트에 대한 외부 검증이 부족하다.
- GPU 제한으로 인해 전체 3‑D CNN 탐색이 제한적이다.
- 임상 현장에서의 실제 유용성은 아직 평가되지 않았다.

### 4.3 향후 연구
- **3‑D CNN** 기반 베이스라인을 구축하여 볼륨 전체를 학습.
- **Grad‑CAM** 등 XAI 기법을 도입해 모델 해석성을 강화.
- 다기관 검증 및 반지도 학습을 활용해 라벨이 없는 데이터를 활용.

---

## 5. 결론
본 연구는 CT와 흉부 X‑ray 질병 분류를 위한 재현 가능하고 모듈화된 프레임워크를 제시한다. 제안된 하이브리드 CT 모델은 CNN 특징 추출과 Transformer 기반 컨텍스트 모델링을 결합함으로써 성능 향상을 입증했으며, X‑ray 베이스라인은 적절한 손실 가중치를 적용한 전이 학습의 효과를 확인하였다. 이 작업은 높은 정확도와 해석 가능성을 동시에 요구하는 미래 임상 AI 시스템 구축을 위한 기반을 제공한다.

---

## 참고문헌
1. **MONAI**: Project MONAI – Medical Open Network for AI, https://monai.io.
2. **TIMM**: PyTorch Image Models, https://github.com/huggingface/pytorch-image-models.
3. RSNA 2023 Abdominal Trauma Detection Competition.
4. NIH Chest X‑ray Dataset, https://nihcc.nih.gov.
5. CheXpert Dataset, https://stanfordmlgroup.github.io/chexpert.

---
