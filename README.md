# 🩺 흉부 X-ray 및 복부 CT 영상 판독 AI 모델 개발
> **의료 영상 처리 딥러닝 포트폴리오** 

본 프로젝트는 의료 영상(흉부 X-ray 및 복부 CT) 데이터를 활용하여 질환을 진단하고 부상 여부를 판별하는 딥러닝 파이프라인 개발 과정을 담고 있습니다. 데이터 전처리부터 모델 아키텍처 구성, 세부 하이퍼파라미터 등 지속적인 성능 튜닝 내역을 문서화하였습니다. 

## 🎯 프로젝트 개요
* **목표:** 의료 이미지를 입력받아 이상 병변 진단 및 손상된 장기 부상 여부를 확률로 제공하는 AI 모델 설계 및 최적화
* **주요 도메인:** 컴퓨터 비전(Computer Vision), 의료 딥러닝(Healthcare & Medical AI)
* **주요 기술 스택:** Python, PyTorch, MONAI, Transformer, Transformer-Encoder, ConvNeXt

---

## 💾 데이터셋
1. **[NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) & [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert)**
   - 두 X-ray 데이터셋을 동일한 환경(DenseNet-121, 전이학습)에 대입하여 분포 차이 및 성능을 교차 평가
2. **[RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)**
   - 3D 복부 CT 입력을 다루며 각 장기(Bowel, Liver, Kidney, Spleen 등)별 상해(Injury) 또는 출혈(Extravasation) 판별

---

## 🛠️ 핵심 모델 아키텍처 (CT 파이프라인)

3D 의료 영상 처리의 컴퓨팅 한계를 해결하고 맥락을 이해시키기 위해 **ConvNeXt + Transformer Encoder** 구조를 고안하였습니다. (V3, V4 모델 전후로 지속 발전)

```text
[입력 데이터] 배치 당 64장의 CT 슬라이스 (Batch, 64, 3, 128, 128 또는 224x224)
↓
1. [Backbone: ConvNeXt-Tiny] → "시각 신경망"
   - 단일 슬라이스의 공간 특징(Feature)을 각각 추출해 고차원 벡터(768~1024차원)로 변환
↓
2. [Position Embedding] → "단층 촬영 스캔 위치 부여"
   - CT 특성에 맞는 Z축(위치 인덱스) 정보를 주입
↓
3. [Transformer Encoder] → "슬라이스 간 종합 분석"
   - 64장의 슬라이스들이 어텐션으로 상호작용하며 장기 부상의 전후 연결(Global Context) 파악
↓
4. [Attention Pooling] → "스포트라이트"
   - 슬라이스들 중 부상 의심 지점이 큰 곳에 집중하여 1개의 '대표 벡터'로 압축
↓
5. [Final Diagnosis Heads] → "최종 전문의 판별"
   - [Suspicion Head]: 복부 부상 유무 확률 도출 (`injury_prob`)
   - [Organ Heads]: 각 장기(Bowel, Liver...)별 디테일 클래시피케이션
   ⭐ 개선점: 전체 부상 확률(`injury_prob`)을 각 장기 판단 결과에 직접 곱해주어 허위 양성(False Positive)을 강력 통제(V4 아키텍처부터 도입)
```

---

## 📈 실험 이력 (Versions & Experiments History)

검증 AUC와 Loss 향상을 위해 점진적인 기법을 적용 및 실험한 기록입니다. 
Loss Class Weight 튜닝, Label Smoothing, LLRD 등 다양한 전략을 도입하며 성능을 비교 분석하였습니다.

### 📊 버전별 모델 성능 요약
| Version | Image Size | Best Epoch | Best AUC | Summary & Strategies |
|:---|:---:|:---:|:---:|:---|
| **v1_2** | 128x128 | 14 | **0.7738** | 2 Epoch 레이어 동결, Augmentation, `any_injury` 가중치. 준수한 밸런스. |
| **v2** | 128x128 | 10 | 0.6793 | 10 Epoch 초반 동결 스케줄, LR & Weight Decay 변경 설정. |
| **v3** | 128x128 | 14 | 0.6892 | **과도기 아키텍처(Transformer 도입)**, 부분 동결/해제 구조 적용. |
| **v4** | 128x128 | 11 | 0.7070 | Forward 진단 구조 전환(총합 부상 확률이 개별 장기 결과에 가중). |
| **v5_2** | 224x224 | 17 | 0.7529 | 해상도를 224로 상향해 해상력 증대 (학습 시간 고려해 점진적 도입). |
| **v6_4** | 224x224 | 22 | 0.7524 | 장기 별 **Custom Class Weight Multipliers** 부여 및 Label Smoothing(0.05). |
| **v7_2** | 224x224 | 12 | 0.7474 | Label Smoothing(0.1) 격상, Gradient Accumulation 도입, 일부 외곡 증강 제거. |
| **v8** | 224x224 | 3 | 0.6951 | Augmentation 완전 제거 실험 (의료영상 특성상 Augmentation의 큰 중요성 반증). |
| **v9** | 224x224 | 5 | 0.7131 | Augmentation 복구, ConvNext에 맞는 LLRD (Layer-wise LR Decay) 시도. |
| **v10** | 224x224 | - | **0.775+** | 고해상도(224)에서 성과가 좋았던 V1의 루틴을 재적용하여 성능(최고점) 극대화. |

---

## 💡 개발 시 주요 문제 해결 (Problem-Solving)

* **3D CT 영상의 한계 자원 극복**
  - 무거운 3D CNN 기반 구조 대신 `2D ConvNeXt-Tiny`로 64개의 각 C-슬라이스를 처리한 후 메모리가 가벼운 `Transformer`로 볼륨 통합(Temporal 정보)하는 전략 채택.
* **불균형 데이터 (Class Imbalance) 대처**
  - 부상 사례(예: Bowel, Extravasation)가 희박한 특성 때문에 `nn.CrossEntropyLoss`의 `weight` 파라미터로 양성 클래스에 강한 패널티(10.0, 5.0 등)를 주어 놓치는 손상을 감소.
* **보수적인 의료 평가 방식 유도 (Over-Prediction Control)**
  - 개별 장기가 부상을 시사하더라도 전체 장기의 부상 확률이 낮다면 이를 기각할 수 있도록 전체 부상 판단용 헤드(Suspicion)와 장기별 헤드(Organ)를 계층형으로 구성하여 False Alarm 예방.
* **일관성 없는 Loss 변동 억제 (Generalization)**
  - 의료 스캔 노이즈로 인해 Loss가 튀는 현상을 막기 위해 `Label smoothing(0.1)`과 적절한 `Dropout`, `Gradient Accumulation` 활용.
