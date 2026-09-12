!pip install monai

import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast  # 속도 및 메모리 최적화
from monai.transforms import (
    Compose, MapTransform, SelectItemsd, RandFlipd, RandAffined,
    RandGridDistortiond, RandGaussianNoised, RandAdjustContrastd,
    RandGaussianSmoothd, Transposed, ToTensord
)
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchmetrics  # AUC 계산
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from tqdm import tqdm
from timm.optim import create_optimizer_v2  # v19: LAYER_DECAY 실제 적용용
import timm
from collections import OrderedDict

# ============================================================
# 설정 & 경로
# ============================================================

# 원본 데이터 기본 경로 (CSV, parquet 읽기용)
BASE_DIR = '/kaggle/input/rsna-2023-abdominal-trauma-detection/'

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS    = 25
LEARNING_RATE = 1e-4
LAYER_DECAY   = 0.8

# v19 = v11(baseline)에서 옵티마이저만 교체한 비교 실험.
# v11은 LAYER_DECAY를 선언만 하고 실제로는 안 썼음(전체 파라미터가 같은 LR) —
# v19는 timm.optim.create_optimizer_v2(..., layer_decay=LAYER_DECAY)로 실제
# 적용해서, epoch 2에 backbone을 통째로 동결 해제할 때 얕은 층은 거의 안
# 흔들리고 깊은 층 위주로만 CT 도메인에 적응하도록 바꿨다. 나머지 설정
# (데이터, augmentation, loss, batch size, epoch 수)은 v11과 전부 동일하게
# 유지해서 이 변경 하나의 효과만 비교할 수 있게 했다.

# 이미 완료된 전처리 npy 파일이 저장된 경로 (새로 생성하지 않음)
SAVE_DIR = '/kaggle/input/rsna-2023-atd-preprocessed-s224/result/'

# 학습 결과 모델 저장 경로
MONAI_MODEL_SAVE_PATH = '/kaggle/working/monai_ct_convnext_v19.pth'

# 이어서 학습할 체크포인트 경로 (빈 문자열이면 처음부터 시작)
MONAI_MODEL_SAVE_PATH_CONTINUE = '/kaggle/input/models/yoodeoksu/ct-v7/pytorch/default/40/monai_ct_convnext_v19_ep4.pth'

IMAGE_TARGET = (64, 224, 224)
NUM_SLICES   = 64

print(f'디바이스: {DEVICE}')

# ============================================================
# 데이터 준비 — any_injury 이진 분류 전용
# ============================================================

# 파일 읽기
train_df = pd.read_csv(f'{BASE_DIR}train_2024.csv')
tags_df  = pd.read_parquet(f'{BASE_DIR}train_dicom_tags.parquet')

# 고유 폴더 경로 추출 및 환자 ID 연결
tags_df['series_path'] = tags_df['path'].str.split('/').str[:-1].str.join('/')
unique_series = tags_df[['PatientID', 'series_path']].drop_duplicates()

data_dicts = []
for idx, row in unique_series.iterrows():
    p_id   = int(row['PatientID'])
    s_path = row['series_path']

    # 해당 환자의 라벨 정보 가져오기
    patient_labels = train_df[train_df['patient_id'] == p_id]
    if len(patient_labels) == 0:
        continue  # 라벨 없는 경우 제외
    labels = patient_labels.iloc[0]

    # any_injury: 1이면 이상, 0이면 정상
    # → [정상 확률, 이상 확률] 형태의 원-핫 벡터
    data_dicts.append({
        'image'     : f'{BASE_DIR}{s_path}',
        'patient_id': p_id,
        'any_injury': np.array([1 - labels['any_injury'], labels['any_injury']]).astype('float32')
    })

# train / val 분리 (환자 단위)
patient_ids = train_df['patient_id'].unique()
train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
train_files = [d for d in data_dicts if d['patient_id'] in train_ids]
val_files   = [d for d in data_dicts if d['patient_id'] in val_ids]

print(f'준비된 데이터 수: {len(data_dicts)}')
print(f'  - Train: {len(train_files)}, Val: {len(val_files)}')

# ============================================================
# 모델 정의 — ConvNeXt-Tiny + Attention Pooling
# ============================================================

class Timm_Model(torch.nn.Module):
    def __init__(self, model_name='convnext_tiny', num_slices=64):
        super().__init__()
        # num_classes=0: 분류 헤드 제거 → 특징 벡터(Feature Vector)만 출력
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # 초기에는 backbone 동결 — 헤드만 학습 (epoch 2부터 해제)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.dim = self.backbone.num_features  # ConvNeXt-Tiny 기준 768

        # 어텐션 풀링: 64장 슬라이스 중 이상 징후 슬라이스에 높은 가중치 부여
        self.attention_net = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 이진 분류 헤드: [정상, 이상] 2-class 출력
        self.suspicion_head = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # x shape: (Batch, 64, 3, 224, 224)
        b, s, c, h, w = x.shape

        chunk_size = 8  # 한 번에 처리할 슬라이스 수 (메모리에 따라 조절)
        all_features = []

        for i in range(0, s, chunk_size):
            x_chunk = x[:, i : i + chunk_size]           # (B, chunk, 3, 224, 224)
            x_chunk = x_chunk.reshape(-1, c, h, w)       # (B*chunk, 3, 224, 224)
            feat_chunk = self.backbone(x_chunk)           # (B*chunk, 768)
            feat_chunk = feat_chunk.view(b, -1, self.dim) # (B, chunk, 768)
            all_features.append(feat_chunk)

        features = torch.cat(all_features, dim=1)         # (B, 64, 768)

        # 어텐션 가중치: 이상 징후 슬라이스 강조
        att_scores  = self.attention_net(features)         # (B, 64, 1)
        att_weights = F.softmax(att_scores, dim=1)         # (B, 64, 1)
        combined    = torch.sum(features * att_weights, dim=1)  # (B, 768)

        return {'any_injury': self.suspicion_head(combined)}


# ============================================================
# MONAI 데이터 파이프라인
# ============================================================

class LoadNpyTransformd(MapTransform):
    '''SAVE_DIR에 저장된 전처리 완료 .npy 파일을 로드하는 Transform.'''
    def __call__(self, data):
        d = dict(data)
        file_path = d['image']
        try:
            img = np.load(file_path)
            if img.shape[-1] == 3:  # (64, 224, 224, 3) → (3, 64, 224, 224)
                img = np.transpose(img, (3, 0, 1, 2))
            d['image'] = torch.from_numpy(img).float()
        except Exception as e:
            print(f'\n❌ npy 파일 로드 실패: {file_path} | 에러: {e}')
            raise e
        return d


def monai_train_pipeline():
    return Compose([
        LoadNpyTransformd(keys=['image']),

        # 공간적 변형 (Spatial Augmentation)
        # spatial_axis: 0=Slices, 1=H, 2=W
        RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),  # 좌우 반전
        RandFlipd(keys=['image'], prob=0.5, spatial_axis=2),  # 상하 반전

        RandAffined(
            keys=['image'],
            prob=0.2,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            translate_range=(10, 10, 10),
            padding_mode='zeros',
            mode='bilinear'
        ),

        # 형태적 변형 (Grid Distortion)
        RandGridDistortiond(
            keys=['image'],
            prob=0.2,
            num_cells=(4, 4, 4),
            distort_limit=(-0.05, 0.05),
            mode='bilinear'
        ),

        # 강도 및 노이즈 (Intensity Augmentation)
        RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.05),
        RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.3)),
        RandGaussianSmoothd(keys=['image'], prob=0.1, sigma_x=(0.5, 1.0)),

        # (3, S, H, W) → (S, 3, H, W): 모델 입력 형식 (Batch, Slices, C, H, W)에 맞게
        Transposed(keys=['image'], indices=(1, 0, 2, 3)),

        # track_meta=False: MONAI MetaTensor를 일반 Tensor로 변환
        # (MetaTensor가 forward()에 그대로 전달되면 CUDA 호환성 오류 발생)
        ToTensord(keys=['image', 'any_injury'], track_meta=False),
        SelectItemsd(keys=['image', 'any_injury'])
    ])


def monai_val_pipeline():
    return Compose([
        LoadNpyTransformd(keys=['image']),
        Transposed(keys=['image'], indices=(1, 0, 2, 3)),
        ToTensord(keys=['image', 'any_injury'], track_meta=False),
        SelectItemsd(keys=['image', 'any_injury'])
    ])

# ============================================================
# 유틸리티
# ============================================================

def process_one_item(item):
    '''원본 시리즈 경로 → 전처리된 npy 파일 경로로 교체. 파일이 없으면 None 반환.'''
    new_item = item.copy()
    s_id = new_item['image'].split('/')[-1]              # 시리즈 ID 추출
    target_path = os.path.join(SAVE_DIR, f'{s_id}.npy')
    if os.path.isfile(target_path):
        new_item['image'] = target_path
        return new_item
    return None


# ============================================================
# 평가 함수
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()
    val_epoch_loss = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='[Validation]'):
            inputs  = batch['image'].to(DEVICE)
            outputs = model(inputs)

            pred   = torch.softmax(outputs['any_injury'], dim=1).cpu()
            target = batch['any_injury'].to(DEVICE)

            loss = criterion(outputs['any_injury'], target)
                loss = loss / ACCUMULATION_STEPS
            val_epoch_loss += loss.item()

            all_preds.append(pred)
            all_trues.append(torch.argmax(batch['any_injury'], dim=1))

    avg_val_loss = val_epoch_loss / len(loader)

    all_preds = torch.cat(all_preds)
    all_trues = torch.cat(all_trues)
    auc_metric = torchmetrics.AUROC(task='multiclass', num_classes=2)
    any_injury_auc = auc_metric(all_preds, all_trues).item()

    return {'any_injury': any_injury_auc}, avg_val_loss


# ============================================================
# 학습 함수
# ============================================================

def train(train_files_preprocessed, val_files_preprocessed,
          train_pipeline, val_pipeline,
          model_save_path):

    # CrossEntropyLoss vs BCEWithLogitsLoss 비교
    # 구분             BCEWithLogitsLoss              CrossEntropyLoss
    # 풀네임           Binary Cross Entropy           (Multiclass) Cross Entropy
    # 주요 목적        이진 분류 (Yes or No)           다중 분류 (A, B, C 중 하나)
    # 출력 노드 수     1개 (0~1 사이의 확률)           N개 (각 클래스별 점수)
    # 활성 함수        Sigmoid (내장됨)                Softmax (내장됨)
    # 타겟 라벨        0.0 또는 1.0 (Float)            0, 1, 2... 인덱스 (Long)
    # 특징             각 타겟이 독립적 (Multi-label)  타겟 간 경쟁 (합이 1)
    # → 이 모델은 [정상, 이상] 2-class Softmax이므로 CrossEntropyLoss 사용

    ACCUMULATION_STEPS = 8
    train_ds = Dataset(data=train_files_preprocessed, transform=train_pipeline)
    val_ds   = Dataset(data=val_files_preprocessed,   transform=val_pipeline)

    # num_workers=0로 축소: RandAffined/RandGridDistortiond가 64슬라이스 3D 볼륨에 대해
    # CPU/RAM을 많이 쓰는데, worker 4개 + prefetch가 겹치면 Kaggle 인스턴스 RAM을 넘겨서
    # 파이썬 예외 없이 커널 자체가 OOM-kill 당하는 경우(DeadKernelError)가 있었다.
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=0, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=False)

    # 모델
    model = Timm_Model(model_name='convnext_tiny').to(DEVICE)
    if torch.cuda.device_count() > 1:
        print('2개의 GPU를 사용합니다.')
        model = nn.DataParallel(model)

    # 손실 함수 — 클래스 불균형 보정 (정상:이상 = 1:5)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 5.0]).to(DEVICE),
        label_smoothing=0.05
    )

    # v19: timm.optim.create_optimizer_v2로 layer_decay 실제 적용
    target_model = model.module if hasattr(model, 'module') else model
    optimizer = create_optimizer_v2(
        target_model, opt='adamw', lr=LEARNING_RATE, weight_decay=1e-5,
        layer_decay=LAYER_DECAY,
    )
    unique_lrs = sorted(set(g['lr'] for g in optimizer.param_groups))
    print(f'[v19 진단] param groups: {len(optimizer.param_groups)}개, '
          f'서로 다른 LR: {len(unique_lrs)}개 → {unique_lrs}')
    if len(unique_lrs) <= 1:
        print('[v19 경고] LR이 전부 같음 = layer_decay가 실제로 안 걸렸을 가능성 높음.')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler    = GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    start_epoch = 0
    history = {
        'train_loss'  : [],
        'val_loss'    : [],
        'auc_avg'     : [],
        'auc_details' : [],
    }

    # ── 체크포인트 로드 ──────────────────────────────────────
    if MONAI_MODEL_SAVE_PATH_CONTINUE and os.path.exists(MONAI_MODEL_SAVE_PATH_CONTINUE):
        print(f'\n🔍 체크포인트 발견: {MONAI_MODEL_SAVE_PATH_CONTINUE}')
        ckpt = torch.load(MONAI_MODEL_SAVE_PATH_CONTINUE, map_location=DEVICE)

        # DataParallel 래퍼 제거 후 로드
        new_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        history = ckpt['history']
        print(f'✅ {start_epoch} 에포크부터 재개합니다.')

    for epoch in range(NUM_EPOCHS):
        if epoch < start_epoch:
            continue

        # ── Backbone 동결 해제 (Epoch 2부터) ─────────────────
        if epoch >= 2:
            target_model = model.module if hasattr(model, 'module') else model
            if not next(target_model.backbone.parameters()).requires_grad:
                print(f'\n🔓 [Epoch {epoch}] Backbone 동결 해제 — 파인튜닝 시작')
                for param in target_model.backbone.parameters():
                    param.requires_grad = True
                if epoch == 2:
                    print('[v19] layer_decay가 정한 층별 LR 그대로 유지 (v11처럼 일괄 덮어쓰기 없음)')

        # ── 학습 ─────────────────────────────────────────────
        model.train()
        train_epoch_loss = 0
        train_loop = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f'Epoch {epoch}/{NUM_EPOCHS} [Train]', leave=False)

        for i, batch in train_loop:
            inputs = batch['image'].to(DEVICE)

            with autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                outputs = model(inputs)
                target  = batch['any_injury'].to(DEVICE)
                if target.dtype != torch.float32:
                    target = target.float()
                loss = criterion(outputs['any_injury'], target)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
    
            train_epoch_loss += loss.item() * ACCUMULATION_STEPS
            train_loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

        avg_train_loss = train_epoch_loss / len(train_loader)

        # ── 검증 ─────────────────────────────────────────────
        auc_results, avg_val_loss = evaluate(model, val_loader, criterion)
        mean_auc = auc_results['any_injury']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['auc_avg'].append(mean_auc)
        history['auc_details'].append(auc_results)

        scheduler.step(avg_val_loss)

        print(f'\n>>> Epoch {epoch} Summary')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'any_injury AUC: {mean_auc:.4f}')
        print('-' * 50)

        # ── 저장 ─────────────────────────────────────────────
        current_save_path = model_save_path.replace('.pth', f'_ep{epoch}.pth')
        torch.save({
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler'   : scaler.state_dict(),
            'history'  : history
        }, current_save_path)
        print(f'✅ 모델 저장: {current_save_path}')

        history_save_path = model_save_path.replace('.pth', '.pkl')
        with open(history_save_path, 'wb') as f:
            pickle.dump(history, f)
        print(f'✅ 히스토리 저장: {history_save_path}')

    return history

# ============================================================
# 실행
# ============================================================

# SAVE_DIR에 존재하는 전처리 파일로 경로 교체 (없는 파일은 제외)
train_files_preprocessed = [r for r in (process_one_item(i) for i in train_files) if r is not None]
val_files_preprocessed   = [r for r in (process_one_item(i) for i in val_files)   if r is not None]

print(f'전처리 파일 매칭: Train {len(train_files_preprocessed)} / Val {len(val_files_preprocessed)}')

# 파이프라인 생성
train_pipeline = monai_train_pipeline()
val_pipeline   = monai_val_pipeline()

# 학습 시작
print('=' * 25, 'Train', '=' * 25)
history = train(
    train_files_preprocessed, val_files_preprocessed,
    train_pipeline, val_pipeline,
    MONAI_MODEL_SAVE_PATH
)

