"""
"확진이 아니라 의심" 관점의 재평가 유틸리티 — 이미 학습된 모델에 재학습 없이
바로 적용 가능(추론만 필요, GPU 거의 안 씀. 무료 Kaggle/Colab에 적합).

지금까지 이 저장소의 CT Screening(v11/v15) 결과는 AUC로만 보고됐다. AUC는
"임계값과 무관하게 전반적으로 얼마나 잘 구분하는가"를 보는 지표라, 실제
배포 시 "어떤 임계값을 쓸지"와 "그 임계값에서 얼마나 놓치는지/얼마나
오탐하는지"는 알려주지 않는다. CADe/CADt(Computer-Aided Detection/Triage)
제품의 실제 관심사는 후자다 — "정밀 검사를 권고해야 하는 케이스를 얼마나
놓치지 않는가(recall/민감도)"가 우선이고, 그 대가로 오탐이 얼마나 늘어나는지를
감수 가능한 수준에서 잡는 것.

사용법: 이진 스크리닝 모델(CT Screening v11/v15처럼 "이상 있음/없음" 단일
헤드로 학습된 모델)의 검증셋 예측 확률(y_scores)과 실제 라벨(y_true)을
그대로 screening_report(y_true, y_scores)에 넣으면 된다
(`ct_screening_inference.py` 참고).

**X-ray DenseNet121에는 이 유틸리티를 적용하지 않는다.** X-ray는 5개
질환을 각각 독립적으로 예측하도록 학습된 모델이라, 그 확률들의 최댓값을
"이상 소견 점수"로 묶는 건 학습 끝난 모델을 사후에 재해석하는 것에 가깝고
CT(v11)처럼 "이상 있음/없음"을 목표로 직접 학습된 모델과 공정하게 비교할
근거가 안 된다(자세한 이유는 image/README.md 한계 섹션 참고). LiTS
세그멘테이션의 케이스 단위 recall은 `ct_organ_segmentation.py`에 이미
내장되어 있어 이 파일이 필요 없다(볼륨 단위 유무 판정이라 확률 스코어 기반
임계값 스윕과는 성격이 달라서 분리해뒀다).
"""
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def threshold_for_recall(y_true, y_scores, target_recall):
    """
    주어진 target_recall(민감도) 이상을 만족하는 가장 "엄격한"(오탐이 적은)
    임계값을 찾는다. 반환: (threshold, 실제 달성 recall, false_positive_rate, precision)
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    thresholds = np.unique(y_scores)[::-1]  # 높은 점수부터 검사 (엄격한 임계값 우선)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0:
        raise ValueError("y_true에 양성(이상 소견) 케이스가 없다 — recall을 정의할 수 없음")

    best = None
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        recall = tp / n_pos
        fpr = fp / n_neg if n_neg > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        if recall >= target_recall:
            best = (t, recall, fpr, precision)
            # 이 임계값부터 목표 recall을 만족하므로, 그중 가장 엄격한(오탐 적은) 것을 채택
            break
    if best is None:
        # 가장 낮은 임계값(전부 양성 예측)으로도 target_recall을 못 채우는 경우
        t = thresholds[-1]
        pred = (y_scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        best = (t, tp / n_pos, fp / n_neg if n_neg > 0 else float("nan"),
                tp / (tp + fp) if (tp + fp) > 0 else float("nan"))
    return best


def screening_report(y_true, y_scores, target_recalls=(0.90, 0.95, 0.99)):
    """
    여러 목표 recall 수준에서 '그 recall을 유지하려면 오탐을 얼마나 감수해야
    하는가'를 표로 보여준다 — "확진이 아니라 의심" 설계를 뒷받침하는 핵심 표.
    """
    print(f"{'목표 recall':>12} | {'threshold':>10} | {'실제 recall':>10} | "
          f"{'오탐률(FPR)':>12} | {'precision':>10}")
    print("-" * 66)
    rows = []
    for target in target_recalls:
        t, recall, fpr, precision = threshold_for_recall(y_true, y_scores, target)
        print(f"{target:>12.2f} | {t:>10.4f} | {recall:>10.3f} | {fpr:>12.3f} | {precision:>10.3f}")
        rows.append({"target_recall": target, "threshold": t, "recall": recall,
                      "fpr": fpr, "precision": precision})
    return rows


if __name__ == "__main__":
    # 사용 예시(더미 데이터) — 실제로는 위 사용법대로 y_true/y_scores를 모델
    # 추론 결과로 채워서 호출할 것.
    rng = np.random.default_rng(0)
    y_true_demo = rng.integers(0, 2, size=500)
    y_scores_demo = np.clip(y_true_demo * 0.4 + rng.normal(0.5, 0.2, size=500), 0, 1)
    screening_report(y_true_demo, y_scores_demo)
