# 흉부 X-ray 및 복부 CT 영상 판독 AI 모델 개발
> **의료 영상 처리 딥러닝 포트폴리오** 

![Best Model (V10) Result](assets/ct_result_history_10.png)
*최고 성능(v10) 모델의 학습 곡선 및 검증 AUC 결과*

본 프로젝트는 의료 영상(흉부 X-ray 및 복부 CT) 데이터를 활용하여 질환을 진단하고 부상 여부를 판별하는 딥러닝 파이프라인 개발 과정을 담고 있습니다. 데이터 전처리부터 모델 아키텍처 구성, 세부 하이퍼파라미터 등 지속적인 성능 튜닝 내역을 문서화하였습니다. 

## 프로젝트 개요
* **목표:** 의료 이미지를 입력받아 이상 병변 진단 및 손상된 장기 부상 여부를 확률로 제공하는 AI 모델 설계 및 최적화
* **주요 도메인:** 컴퓨터 비전(Computer Vision), 의료 딥러닝(Healthcare & Medical AI)
* **주요 기술 스택:** Python, PyTorch, MONAI, Transformer, Transformer-Encoder, ConvNeXt

---

## 데이터셋
1. **[NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) & [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert)**
   - 두 X-ray 데이터셋을 동일한 환경(DenseNet-121, 전이학습)에 대입하여 분포 차이 및 성능을 교차 평가
2. **[RSNA 2023 Abdominal Trauma Detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)**
   - 3D 복부 CT 입력을 다루며 각 장기(Bowel, Liver, Kidney, Spleen 등)별 상해(Injury) 또는 출혈(Extravasation) 판별

---

## 핵심 모델 아키텍처 (CT 파이프라인 - Best Version: V10)

3D 의료 영상 처리의 효율성을 극대화하기 위해 **ConvNeXt + Attention Pooling** 구조를 최종 채택하였습니다. 
(복잡한 Transformer 구조 대신, 핵심 특징에 집중하는 Pooling 방식을 고해상도 학습 루틴과 결합하여 최고 성능을 달성한 버전입니다.)

```text
[입력 데이터] 배치 당 64장의 CT 슬라이스 (Batch, 64, 3, 224, 224)
↓
1. [Backbone: ConvNeXt-Tiny] → "시각 신경망"
   - 단일 슬라이스의 공간 특징(Feature)을 각각 추출해 고차원 벡터로 변환
↓
2. [Attention Pooling] → "심사위원 (핵심 슬라이스 포착)"
   - 64장의 슬라이스 특징 중 부상 의심 지점이 큰 곳에 높은 가중치를 부여
   - 가중합을 통해 64개의 벡터를 1개의 '최종 대표 벡터'로 압축
↓
3. [Parallel Diagnosis Heads] → "최종 전문의 판별"
   - [Suspicion Head]: 전체 복부 부상 유무 확률 도출 (`any_injury`)
   - [Organ Heads]: 각 장기(Bowel, Liver...)별 정밀 진단 수행
   기술적 의사결정: V3, V4에서 시도했던 Transformer 및 Gating(가중치 곱셈) 로직보다, 
     고해상도 환경에서 Attention Pooling 기반의 독립적 Head 구조가 더 안정적인 AUC(0.775+)를 
     보임을 확인하여 최종 모델로 확정.
```

---

## 실험 이력 (Versions & Experiments History)

검증 AUC와 Loss 향상을 위해 점진적인 기법을 적용 및 실험한 기록입니다. 
Loss Class Weight 튜닝, Label Smoothing, LLRD 등 다양한 전략을 도입하며 성능을 비교 분석하였습니다.

### 버전별 모델 성능 요약
| Version | Image Size | Best Epoch | Best AUC | Summary & Strategies |
|:---|:---:|:---:|:---:|:---|
| **v1_2** | 128x128 | 14 | **0.7738** | 2 Epoch 레이어 동결, Augmentation, `any_injury` 가중치. 준수한 밸런스. |
| **v2** | 128x128 | 10 | 0.6793 | 10 Epoch 초반 동결 스케줄, LR & Weight Decay 변경 설정. |
| **v3** | 128x128 | 14 | 0.6892 | **실험적 아키텍처(Transformer 도입)**, 부분 동결/해제 구조 적용. |
| **v4** | 128x128 | 11 | 0.7070 | **Gating 구조 실험**(부상 확률을 개별 결과에 가중치로 곱해주는 시도). |
| **v5_2** | 224x224 | 17 | 0.7529 | 해상도를 224로 상향해 해상력 증대 (학습 시간 고려해 점진적 도입). |
| **v6_4** | 224x224 | 22 | 0.7524 | 장기 별 **Custom Class Weight Multipliers** 부여 및 Label Smoothing(0.05). |
| **v7** | 224x224 | 12 | 0.7474 | Label Smoothing(0.1) 격상, Gradient Accumulation 도입, 일부 외곡 증강 제거. |
| **v8** | 224x224 | 3 | 0.6951 | Augmentation 완전 제거 실험 (의료영상 특성상 Augmentation의 큰 중요성 반증). |
| **v9** | 224x224 | 5 | 0.7131 | Augmentation 복구, ConvNext에 맞는 LLRD (Layer-wise LR Decay) 시도. |
| **v10** | 224x224 | - | **0.775+** | **Best Full-Model**. 고해상도(224)에서 v1의 강력한 Pooling 루틴을 재적용하여 성능 극대화. |
| **v11** | 224x224 | 16 | **0.825+** | **부상 여부 판별 특화 모델**. 개별 장기가 아닌 전체 부상 유무만 집중 학습. |
| **v12** | 224x224 | 10 | **0.8047** | `BCEWithLogitsLoss` 및 `Pos-Weight(5.0)` 도입 실험. CE 대비 학습 불안정성(shaking) 증가 및 AUC 소폭 하락 확인. |

---

## 상세 실험 이력 (Detailed Changes)

전체 모델의 점진적 발전 과정을 코드를 분석하여 정리하였습니다.

| 버전 | 주요 변경 사항 | 결과 및 분석 |
|:---:|:---|:---|
| **v1-v2** | s128, Backbone 동결 기간 조정 | v1의 2 epoch 동결이 v2의 10 epoch 보다 학습 효율 및 AUC 면에서 우수함 확인. |
| **v3** | **Transformer Encoder** 도입 | 3D 볼륨 내의 슬라이스 간 관계 학습 시도. 초기에는 AUC가 높지 않아 구조적 한계 확인. |
| **v4** | **Gating (Weighting)** 구조 시도 | 부상 확률을 각 장기 헤드에 곱해주는 계층형 구조 시뮬레이션. 오탐 억제 효과 확인. |
| **v5** | 이미지 크기 상향 (128 → 224) | 고해상도 입력으로 미세 병변 파악 능력 개선, AUC 0.75 선 진입. |
| **v6-v7** | Class Weight & Label Smoothing | 불균형 데이터 해소를 위해 `Bowel`, `Extravasation` 등에 가중치 부여 및 변동성 억제. |
| **v8-v9** | Augmentation Ablation Study | 증강 제거(v8) 시 성능 대폭 하락. v9에서 복구 및 **LLRD** 전략 도입하여 안정화. |
| **v10** | **Best Full-Model** 루틴 재정립 | **V3, V4의 복잡한 구조보다 단순한 Attention Pooling이 고해상도에서 더 강력함**을 증명. |
| **v11** | Injury-Only Monitoring | '병명'이 아닌 '위급성(부상 유무)'에만 집중하여 분류 성능을 0.82 이상으로 끌어올림. |
| **v12** | Loss Function 최적화 | Multi-class CE에서 **BCEWithLogitsLoss**로 전환. 수치상 v11 대비 성능 향상은 없으나, 최적 임계값 탐색 데이터 확보. |

---

## 고찰

*   **2.5D 하이브리드 아키텍처**: 무거운 3D CNN 기반 구조 대신 `ConvNeXt(Spatial)` 처리 속도 개선. 
*   **임상적 우선순위 반영**: 단순 정확도가 아닌, 위급 환자를 놓치지 않기 위한 가중 손실 함수(`Any_Injury Priority`) 설계 역량 증명.
*   **데이터 과학적 엄밀성**: v8 실험(Ablation Study)을 통해 의료 도메인에서 데이터 증강(Augmentation)이 모델 일반화에 미치는 영향을 수치로 입증.

## 향후 계획 (Future Works)
*   **v12 성능 분석 완료**: BCE Loss 조정 시 Loss Shaking 현상 및 AUC 정체 확인. 
*   **병명 검출** : v11(AUC 0.812)의 안정성을 기반으로 장기별 병명 분류 학습 추가 진행.