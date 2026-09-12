import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timm
import os

from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose,
    MapTransform,
    Transposed,
    ToTensord,
    SelectItemsd
)

from monai.data import DataLoader, Dataset

# =========================================================
# 1. 환경 설정
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ORGANS = [
    'bowel',
    'extravasation',
    'liver',
    'kidney',
    'spleen'
]

BASE_DIR = '/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection/'
SAVE_DIR = '/kaggle/input/datasets/yoodeoksu/rsna-2023-atd-preprocessed-s224/result/'

V15_THRESH = 0.5


# =========================================================
# 2. Pickle 호환성 패치
# =========================================================

class DummyStub:
    def __getattr__(self, name):
        return DummyStub()

    def __call__(self, *args, **kwargs):
        return None


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "torch.utils.serialization" in module:
            return DummyStub

        return super().find_class(module, name)


class SafePickleModule:
    @staticmethod
    def load(file, **kwargs):
        return RenameUnpickler(file, **kwargs).load()

    Unpickler = RenameUnpickler


sys.modules["torch.utils.serialization"] = DummyStub()
sys.modules["torch.utils.serialization.config"] = DummyStub()


# =========================================================
# 3. 모델 구조
# =========================================================

class RSNAModel(nn.Module):

    def __init__(self, model_type='v17'):

        super().__init__()

        self.model_type = model_type

        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=True,
            num_classes=0,
            drop_path_rate=0.1
        )

        # Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.dim = self.backbone.num_features

        # Attention Pooling
        self.attention_net = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # -----------------------------
        # v15 : Screening
        # -----------------------------
        if model_type == 'v15':

            self.suspicion_head = nn.Sequential(
                nn.Linear(self.dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )

        # -----------------------------
        # v17 : Organ Diagnosis
        # -----------------------------
        else:

            self.heads = nn.ModuleDict({

                "bowel": nn.Linear(self.dim, 2),

                "extravasation": nn.Linear(self.dim, 2),

                "liver": nn.Linear(self.dim, 3),

                "kidney": nn.Linear(self.dim, 3),

                "spleen": nn.Linear(self.dim, 3)
            })

    def forward(self, x):

        b, s, c, h, w = x.shape

        chunk_size = 8

        all_features = []

        # =================================================
        # Chunk Inference
        # =================================================

        for i in range(0, s, chunk_size):

            x_chunk = x[:, i:i+chunk_size]

            x_chunk = x_chunk.reshape(-1, c, h, w)

            feat_chunk = self.backbone(x_chunk)

            feat_chunk = feat_chunk.view(b, -1, self.dim)

            all_features.append(feat_chunk)

        features = torch.cat(all_features, dim=1)

        # =================================================
        # Attention Pooling
        # =================================================

        att_scores = self.attention_net(features)

        att_weights = F.softmax(att_scores, dim=1)

        combined = torch.sum(features * att_weights, dim=1)

        # =================================================
        # Output
        # =================================================

        if self.model_type == 'v15':

            return self.suspicion_head(combined)

        else:

            return {
                name: self.heads[name](combined)
                for name in self.heads
            }


# =========================================================
# 4. 모델 로드
# =========================================================

def load_rsna_model(path, model_type):

    model = RSNAModel(model_type=model_type).to(DEVICE)

    ckpt = torch.load(
        path,
        map_location='cpu',
        pickle_module=SafePickleModule,
        weights_only=False
    )

    state_dict = ckpt.get(
        'model_ema',
        ckpt.get('model', ckpt)
    )

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():

        name = k[7:] if k.startswith('module.') else k

        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    return model


# =========================================================
# 5. 데이터 로드
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

        Transposed(
            keys=["image"],
            indices=(1, 0, 2, 3)
        ),

        ToTensord(
            keys=["image"] + ORGANS + ["any_injury"],
            track_meta=False
        ),

        SelectItemsd(
            keys=["image"] + ORGANS + ["any_injury"]
        )
    ])


def process_one_item(item):

    new_item = item.copy()

    s_id = new_item['image'].split("/")[-1]

    target_path = os.path.join(
        SAVE_DIR,
        f"{s_id}.npy"
    )

    if os.path.isfile(target_path):

        new_item['image'] = target_path

        return new_item

    return None


# =========================================================
# 6. Validation 데이터 생성
# =========================================================

train_df = pd.read_csv(f'{BASE_DIR}train_2024.csv')

tags_df = pd.read_parquet(
    f'{BASE_DIR}train_dicom_tags.parquet'
)

unique_series = tags_df[['PatientID', 'path']].copy()

unique_series['series_path'] = (
    unique_series['path']
    .str.split('/')
    .str[:-1]
    .str.join('/')
)

unique_series = unique_series[
    ['PatientID', 'series_path']
].drop_duplicates()

data_dicts = []

for _, row in unique_series.iterrows():

    p_id = int(row['PatientID'])

    labels = train_df[
        train_df['patient_id'] == p_id
    ]

    if len(labels) == 0:
        continue

    l = labels.iloc[0]

    data_dicts.append({

        "image": f"{BASE_DIR}{row['series_path']}",

        "patient_id": p_id,

        "bowel":
            l[['bowel_healthy', 'bowel_injury']]
            .values.astype("float32"),

        "extravasation":
            l[['extravasation_healthy',
               'extravasation_injury']]
            .values.astype("float32"),

        "liver":
            l[['liver_healthy',
               'liver_low',
               'liver_high']]
            .values.astype("float32"),

        "kidney":
            l[['kidney_healthy',
               'kidney_low',
               'kidney_high']]
            .values.astype("float32"),

        "spleen":
            l[['spleen_healthy',
               'spleen_low',
               'spleen_high']]
            .values.astype("float32"),

        "any_injury":
            np.array([
                1 - l['any_injury'],
                l['any_injury']
            ]).astype("float32")
    })


_, val_ids = train_test_split(
    train_df['patient_id'].unique(),
    test_size=0.2,
    random_state=42
)

val_files_preprocessed = [

    r for r in [

        process_one_item(d)

        for d in data_dicts

        if d['patient_id'] in val_ids

    ]

    if r is not None
]


# =========================================================
# 7. 모델 로드
# =========================================================

v15_path = "/kaggle/input/models/ejrtnyoo/ct-model/pytorch/default/2/monai_ct_convnext_v15_ep28.pth"

v17_path = "/kaggle/input/models/ejrtnyoo/ct-model/pytorch/default/2/monai_ct_convnext_v17_ep29.pth"

model_v15 = load_rsna_model(v15_path, 'v15')

model_v17 = load_rsna_model(v17_path, 'v17')


# =========================================================
# 8. DataLoader
# =========================================================

val_loader = DataLoader(

    Dataset(
        data=val_files_preprocessed,
        transform=monai_val_pipeline()
    ),

    batch_size=1,
    shuffle=False
)


# =========================================================
# 9. 3-Stage Medical Pipeline
# =========================================================

results = []

VERIFY_THRESH = 0.40

print(f"\n📊 {len(val_files_preprocessed)}개 샘플 추론 시작")

for batch in tqdm(val_loader):

    inputs = batch["image"].to(DEVICE)

    gt_any = torch.argmax(
        batch["any_injury"],
        dim=1
    ).item()

    gt_organs = {

        o: (
            1
            if torch.argmax(batch[o], dim=1).item() > 0
            else 0
        )

        for o in ORGANS
    }

    with torch.no_grad():

        # =================================================
        # STEP 1
        # Screening
        # =================================================

        out15 = model_v15(inputs)

        prob15 = F.softmax(
            out15,
            dim=1
        )[0, 1].item()

        is_flagged = int(prob15 >= V15_THRESH)

        # =================================================
        # STEP 2
        # Organ Diagnosis
        # =================================================

        prob17_dict = {
            o: 0.0 for o in ORGANS
        }

        final_injury = 0

        verification_score = 0.0

        if is_flagged:

            out17 = model_v17(inputs)

            for o in ORGANS:

                logits = out17[o]

                p = F.softmax(logits, dim=1)[0]

                if logits.shape[1] == 3:

                    prob17_dict[o] = (
                        p[1] + p[2]
                    ).item()

                else:

                    prob17_dict[o] = p[1].item()

            # =================================================
            # STEP 3
            # Verification
            # =================================================

            verification_score = max(
                prob17_dict.values()
            )

            if verification_score >= VERIFY_THRESH:

                final_injury = 1

            else:

                final_injury = 0

    # =====================================================
    # Save Result
    # =====================================================

    result = {

        "gt_any": gt_any,

        # STEP 1
        "prob_v15": prob15,

        "is_flagged": is_flagged,

        # STEP 3
        "verification_score": verification_score,

        "final_injury": final_injury
    }

    for o in ORGANS:

        result[f"gt_{o}"] = gt_organs[o]

        result[f"prob_{o}"] = prob17_dict[o]

    results.append(result)


df = pd.DataFrame(results)

# =========================================================
# 10. AUC 계산
# =========================================================

print("\n" + "="*60)
print("🏆 3-Stage Medical Pipeline Result")
print("="*60)

# ---------------------------------------------------------
# STEP 1 AUC
# ---------------------------------------------------------

auc_step1 = roc_auc_score(
    df['gt_any'],
    df['prob_v15']
)

print(f"\n[STEP 1] Screening AUC")
print(f"AUC : {auc_step1:.4f}")

# ---------------------------------------------------------
# STEP 2 Organ AUC
# ---------------------------------------------------------

print(f"\n[STEP 2] Organ Diagnosis AUC")

organ_aucs = {}

for o in ORGANS:

    if len(df[f"gt_{o}"].unique()) > 1:

        auc = roc_auc_score(
            df[f"gt_{o}"],
            df[f"prob_{o}"]
        )

        organ_aucs[o] = auc

        print(f"{o:15s} : {auc:.4f}")

# ---------------------------------------------------------
# STEP 3 Final Pipeline AUC
# ---------------------------------------------------------

auc_pipeline = roc_auc_score(
    df['gt_any'],
    df['verification_score']
)

print(f"\n[STEP 3] Final Verification AUC")
print(f"AUC : {auc_pipeline:.4f}")

# =========================================================
# 11. Visualization
# =========================================================

plt.figure(figsize=(15, 6))

# ---------------------------------------------------------
# Step 1 ROC
# ---------------------------------------------------------

plt.subplot(1, 2, 1)

fpr, tpr, _ = roc_curve(
    df['gt_any'],
    df['prob_v15']
)

plt.plot(
    fpr,
    tpr,
    lw=3,
    color='black',
    label=f'v15 Screening (AUC={auc_step1:.3f})'
)

plt.plot([0,1], [0,1], 'k--', alpha=0.5)

plt.title("Step 1 : Injury Screening")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()


# ---------------------------------------------------------
# Step 2 ROC
# ---------------------------------------------------------

plt.subplot(1, 2, 2)

colors = sns.color_palette("Set2", len(ORGANS))

for i, o in enumerate(ORGANS):

    if len(df[f"gt_{o}"].unique()) > 1:

        auc = roc_auc_score(
            df[f"gt_{o}"],
            df[f"prob_{o}"]
        )

        fpr, tpr, _ = roc_curve(
            df[f"gt_{o}"],
            df[f"prob_{o}"]
        )

        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            label=f'{o} (AUC={auc:.3f})'
        )

plt.plot([0,1], [0,1], 'k--', alpha=0.5)

plt.title("Step 2 : Organ Diagnosis")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()

# ---------------------------------------------------------
# STEP 3 ROC
# ---------------------------------------------------------

plt.figure(figsize=(7,7))

fpr, tpr, _ = roc_curve(
    df['gt_any'],
    df['verification_score']
)

plt.plot(
    fpr,
    tpr,
    lw=3,
    color='red',
    label=f'Final Pipeline (AUC={auc_pipeline:.3f})'
)

plt.plot([0,1], [0,1], 'k--', alpha=0.5)

plt.title("Step 3 : Final Verification")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.show()