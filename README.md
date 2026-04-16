# 흉부 X-ray 및 복부 CT 영상 판독 AI 모델 개발
> **의료 영상 처리 딥러닝 포트폴리오** 

![Best Model (V11) Result](assets/ct_result_history_11.png)
*최고 성능(v11, AUC 0.825+) 모델의 부상(Injury) 판별 검증 성능*

본 프로젝트는 의료 영상(흉부 X-ray 및 복부 CT) 데이터를 활용하여 질환을 진단하고 부상 여부를 판별하는 딥러닝 파이프라인 개발 과정을 담고 있습니다. 데이터 전처리부터 모델 아키텍처 구성, 세부 하이퍼파라미터 등 총 16여 차례의 점진적 실험을 통한 성능 튜닝 내역을 상세히 문서화하였습니다.

## 프로젝트 개요
* **목표:** 의료 이미지를 입력받아 부상 여부를 확률로 제공하는 AI 모델 설계
* **주요 도메인:** 컴퓨터 비전(Computer Vision), 의료 딥러닝(Healthcare & Medical AI)
* **주요 기술 스택:** Python, PyTorch, MONAI, Transformer-Encoder, ConvNeXt, EMA, LLRD

---

## 데이터셋
1. **[NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) & [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert)**
   - 두 X-ray 데이터셋을 동일한 환경(DenseNet-121, 전이학습)에서 교차 평가하여 데이터 분포 차이에 따른 도메인 일반화 성능 확인.
2. **[RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)**
   - 3D 복부 CT 볼륨 데이터를 활용하여 상해(Injury) 여부 판별.

---

## 핵심 모델 아키텍처 (CT 파이프라인 - Best Performance Version: V11)

3D CT 스캔 데이터를 여러 개의 2D 슬라이스로 처리하고, 그중 부상이 의심되는 핵심 슬라이스에 집중(Attention)하여 최종적으로 부상 여부(any_injury)를 판별하는 구조입니다. V11은 여러 장기 분석보다 '전체 부상 유무' 판별에 집중하여 최고 성능(AUC 0.825+)을 달성한 모델입니다.

### 1. Model Implementation (PyTorch)

```python
class Timm_Model(torch.nn.Module):
    def __init__(self, model_name='convnext_tiny', num_slices=64):
        super().__init__()
        # 1. Backbone: ConvNeXt-Tiny (Pretrained)
        # 이미지의 핵심 특징 정보(Feature Vector)를 추출 (Tiny 기준 768차원)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.dim = self.backbone.num_features 

        # 2. Attention Pooling: 64장 중 수상한 슬라이스를 골라내는 '심사위원'
        self.attention_net = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 3. Suspicion Head: "부상 여부 탐지" 전용 헤드
        self.suspicion_head = nn.Sequential(
            nn.Linear(self.dim, 256),  # 768 -> 256 압축
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)         # 최종 [정상, 부상] 확률 도출
        )

    def forward(self, x):
        # x shape: (Batch, 64, 3, 224, 224)
        b, s, c, h, w = x.shape
        
        # 슬라이스별 특징 추출 (Backbone)
        x = x.view(-1, c, h, w) 
        features = self.backbone(x) 
        features = features.view(b, s, self.dim) # (Batch, 64, 768)

        # Attention Pooling으로 결정적 증거(핵심 슬라이스) 포착
        att_scores = self.attention_net(features) # (B, 64, 1)
        att_weights = F.softmax(att_scores, dim=1)
        
        # 가중 합산을 통해 시퀀스를 1개의 대표 벡터로 압축
        combined = torch.sum(features * att_weights, dim=1) # (B, 768)

        # 최종 진단
        return {"any_injury": self.suspicion_head(combined)}
```

### 2. 주요 아키텍처 특징
*   **ConvNeXt-Tiny Backbone**: 최신 CNN 구조를 통해 2D 슬라이스의 공간 특징을 정밀하게 추출합니다.
*   **Attention Pooling**: 64장의 슬라이스 중 부상과 무관한 배경 슬라이스의 노이즈를 억제하고, 실제 병변이 포함된 '결정적 증거' 슬라이스에 가중치를 부여합니다.
*   **Binary Focus**: 개별 장기별 분류보다 `any_injury`라는 핵심 문제에 집중함으로써, 위급 환자 분류 모델로서의 실무적 효용성을 극대화했습니다.
*   **Class Imbalance Handling**: 손실 함수에 높은 가중치(`injury_weight = [1.0, 5.0]`)를 부여하여 부상 환자를 놓치지 않도록(Recall 향상) 설계되었습니다.


---

## 버전별 모델 성능 요약
| Version | Image Size | Best Epoch | Best AUC | Summary & Strategies |
|:---|:---:|:---:|:---:|:---|
| **v1_2** | 128x128 | 14 | 0.7738 | Baseline. 2 Epoch 동결, `any_injury` 가중치 손실함수 적용. |
| **v2** | 128x128 | 10 | 0.6793 | 초반 10 Epoch 장기 동결 실험. 학습 속도 저하 확인. |
| **v3** | 128x128 | 14 | 0.6892 | Transformer Encoder 도입. 슬라이스 간 관계 학습 시도. |
| **v4** | 128x128 | 11 | 0.7070 | Gating 구조 도입. 부상 확률을 장기 헤드에 곱해 오탐 억제 시도. |
| **v5_2** | 224x224 | 17 | 0.7529 | 복부 CT 정밀 판독을 위해 입력 해상도 상향. 성능 비약적 상승. |
| **v6_4** | 224x224 | 22 | 0.7524 | 장기 별 가중치 및 Label Smoothing 적용. |
| **v7** | 224x224 | 12 | 0.7474 | Gradient Accumulation 도입 및 규제 강화. |
| **v8** | 224x224 | 3 | 0.6951 | Augmentation 완전 제거 시 성능 폭락 확인 (강건성 입증). |
| **v9** | 224x224 | 5 | 0.7131 | 증강 복구 및 LLRD (Layer-wise LR Decay) 전략 최초 도입. |
| **v10** | 224x224 | - | 0.7791 | 고해상도 Attention 루틴 최적화. |
| **v11** | 224x224 | 16 | 0.8253 | 부상 유무 판별에 집중하여 최고 성능 달성. |
| **v12** | 224x224 | 10 | 0.8047 | BCEWithLogitsLoss 전환 실험. 수렴 속도 개선 및 최적 임계값 확보. |
| **v13** | 224x224 | 9 | 0.7842 | Transformer 재도입 실험. 아키텍처 단순화 버전 대비 효율성 한계 확인. |
| **v14_2**| 224x224 | 40 | 0.7660 | 현대적 훈련 기법 이식. EMA, Stochastic Depth, Warm Restart 적용. |
| **v15** | 224x224 | 40 | 0.7843 | 규제 최적화 및 EMA 적용으로 가장 탄탄한 검증 곡선 획득. |
| **v16** | 224x224 | - | - | LLRD 제거를 통한 학습 수렴 속도 및 성능 민감도 분석 진행 중. |

---

## 상세 실험 이력

전체 모델의 점진적 발전 과정을 분석한 핵심 기술 요약입니다.

| 버전 | 주요 변경 사항 | 결과 및 분석 |
|:---:|:---|:---|
| **v1-v3** | Baseline 셋업 및 구조적 시도 | 기본적인 2.5D 구조에서 Transformer 기반 3D 컨텍스트 학습 시도 (v3). |
| **v4** | Gating (Filtering) 시뮬레이션 | 전체 부상 확률을 필터로 활용하여 장기별 오탐을 물리적으로 억제하는 아키텍처 설계. |
| **v5-v7** | 고해상도 학습 및 불균형 해소 | 224 해상도 상향 및 클래스별 가중치(`Bowel`, `Extravasation` 등) 튜닝으로 실전 성능 확보. |
| **v8-v9** | 데이터 강건성(Robustness) 검증 | 증강 제거 실험(v8)을 통해 의료 도메인 내 데이터 증강의 필수 가치를 수치로 반증. |
| **v10-v11** | 성능 임계치 돌파 | V11에서 부상 판별에만 집중했을 때 AUC 0.82 초과 달성. 의사 결정 우선순위의 중요성 확인. |
| **v14** | A ConvNet for the 2020s 논문 참고 | EMA (Exponential Moving Average) 와 LLRD 적용. 요동치는 학습 곡선을 안정화. |
| **v15** | 아키텍처 정제 및 최종 안정화 | Transformer를 걷어내고 순수 Attention Pooling + EMA 조합으로 가장 신뢰도 높은 모델 완성. |
| **v16** | 하이퍼파라미터 민감도 분석 | LLRD 제거 실험을 통해 백본 학습률 제어가 일반화에 미치는 영향 최종 검증 중. |

---

## 회고

*   **2.5D 하이브리드 아키텍처 최적화**: 연산량이 큰 3D 모델 대신, ConvNeXt와 Attention 기법을 결합하여 실시간성(Latency)과 정확도를 동시에 확보한 경험이었습니다.
*   **최신 기술**: 최신 기술을 적용 한다 해서 성능이 올라가는 것은 아니다. 
*   **임상적 도메인 지식 반영**: '맹목적 분류'가 아닌, '위급 환자 누락 방지'라는 임상 목적에 맞춰 Loss 가중치를 조절하고 Gating 구조를 설계하며 데이터 사이언티스트로서 거시적 안목을 가질 수 있었습니다.