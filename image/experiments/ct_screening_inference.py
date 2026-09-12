"""
v11(Screening) 단독 추론 — Kaggle 전용 (RSNA 원본 CT 데이터가 Kaggle에만 있음).

이전 `2step.py`(삭제됨)는 v15+v17 3단계 파이프라인 전체를 검증하는 용도였다.
지금은 v15보다 실측 AUC가 높은 v11로 Screening 모델을 교체했고(image/README.md
"왜 v15가 아니라 v11인가" 참고), 목적도 "파이프라인 성능 확인"에서 "확진이
아니라 의심 철학을 recall 관점에서 재평가"로 바뀌어서, v11 하나만 가볍게
추론하고 바로 `screening_eval.py`로 넘기는 스크립트로 다시 짰다.

실행 (Kaggle):
    !pip install monai timm
    (Kaggle Model에 v11 가중치를 올려두거나, assets/monai_ct_convnext_v11_ep16.pth를
    이 노트북 환경에 업로드한 뒤 MODEL_PATH를 그 경로로 수정)
"""
import sys
import pickle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import timm
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from monai.transforms import Compose, MapTransform, Transposed, ToTensord, SelectItemsd
from monai.data import DataLoader, Dataset

from screening_eval import screening_report

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection/"
SAVE_DIR = "/kaggle/input/datasets/yoodeoksu/rsna-2023-atd-preprocessed-s224/result/"
MODEL_PATH = "/kaggle/input/models/ejrtnyoo/ct-model/pytorch/default/2/monai_ct_convnext_v11_ep16.pth"


# =========================================================
# 모델 구조 (2step.py의 RSNAModel에서 v15 전용 부분만 남김 — v11도 같은
# "suspicion_head" 이진 스크리닝 구조)
# =========================================================

class ScreeningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=True, num_classes=0, drop_path_rate=0.1
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.dim = self.backbone.num_features

        self.attention_net = nn.Sequential(
            nn.Linear(self.dim, 256), nn.Tanh(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
        self.suspicion_head = nn.Sequential(
            nn.Linear(self.dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 2)
        )

    def forward(self, x):
        b, s, c, h, w = x.shape
        chunk_size = 8
        all_features = []
        for i in range(0, s, chunk_size):
            x_chunk = x[:, i:i + chunk_size].reshape(-1, c, h, w)
            feat_chunk = self.backbone(x_chunk).view(b, -1, self.dim)
            all_features.append(feat_chunk)
        features = torch.cat(all_features, dim=1)

        att_scores = self.attention_net(features)
        att_weights = F.softmax(att_scores, dim=1)
        combined = torch.sum(features * att_weights, dim=1)
        return self.suspicion_head(combined)


def load_model(path):
    model = ScreeningModel().to(DEVICE)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
    new_state_dict = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items()
    )
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


# =========================================================
# 데이터 (2step.py와 동일 — 같은 random_state=42 split을 써야
# 원래 학습 때의 validation set과 동일한 케이스로 비교 가능)
# =========================================================

class LoadNpyTransformd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        img = np.load(d["image"])
        if img.shape[-1] == 3:
            img = np.transpose(img, (3, 0, 1, 2))
        d["image"] = torch.from_numpy(img).float()
        return d


def monai_val_pipeline():
    return Compose([
        LoadNpyTransformd(keys=["image"]),
        Transposed(keys=["image"], indices=(1, 0, 2, 3)),
        ToTensord(keys=["image", "any_injury"], track_meta=False),
        SelectItemsd(keys=["image", "any_injury"]),
    ])


def build_val_files():
    import os
    train_df = pd.read_csv(f"{BASE_DIR}train_2024.csv")
    tags_df = pd.read_parquet(f"{BASE_DIR}train_dicom_tags.parquet")

    unique_series = tags_df[["PatientID", "path"]].copy()
    unique_series["series_path"] = unique_series["path"].str.split("/").str[:-1].str.join("/")
    unique_series = unique_series[["PatientID", "series_path"]].drop_duplicates()

    data_dicts = []
    for _, row in unique_series.iterrows():
        p_id = int(row["PatientID"])
        labels = train_df[train_df["patient_id"] == p_id]
        if len(labels) == 0:
            continue
        l = labels.iloc[0]
        data_dicts.append({
            "image": f"{BASE_DIR}{row['series_path']}",
            "patient_id": p_id,
            "any_injury": np.array([1 - l["any_injury"], l["any_injury"]]).astype("float32"),
        })

    _, val_ids = train_test_split(train_df["patient_id"].unique(), test_size=0.2, random_state=42)

    val_files = []
    for d in data_dicts:
        if d["patient_id"] not in val_ids:
            continue
        s_id = d["image"].split("/")[-1]
        target_path = os.path.join(SAVE_DIR, f"{s_id}.npy")
        if os.path.isfile(target_path):
            d = d.copy()
            d["image"] = target_path
            val_files.append(d)
    return val_files


def main():
    print(f"Device: {DEVICE}")
    model = load_model(MODEL_PATH)
    val_files = build_val_files()
    val_loader = DataLoader(
        Dataset(data=val_files, transform=monai_val_pipeline()), batch_size=1, shuffle=False
    )
    print(f"{len(val_files)}개 검증 케이스 추론 시작")

    rows = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch["image"].to(DEVICE)
            gt_any = torch.argmax(batch["any_injury"], dim=1).item()
            prob = F.softmax(model(inputs), dim=1)[0, 1].item()
            rows.append({"gt_any": gt_any, "prob_v11": prob})

    df = pd.DataFrame(rows)
    df.to_csv("ct_v11_screening_predictions.csv", index=False)
    print("ct_v11_screening_predictions.csv 저장 완료")

    print("\n" + "=" * 60)
    print("STEP 1 Screening(v11) - 의심(recall) 우선 재평가")
    print("=" * 60)
    screening_report(df["gt_any"], df["prob_v11"])


if __name__ == "__main__":
    main()
