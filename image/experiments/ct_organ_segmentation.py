"""
복부 CT 간(liver) + 종양(tumor) Segmentation — Kaggle/Colab GPU용.

RSNA 트라우마 분류(2step.py, submission-result.ipynb)가 "장기가 다쳤나 안
다쳤나"만 분류했다면, 이건 간의 정확한 3D 윤곽(+ 종양 영역)을 복셀(voxel)
단위로 찾아낸다. 같은 문제(복부 CT, 5개 진단 장기 중 하나인 간)를 다른
태스크로 확장한 것이라 RSNA 작업과 자연스럽게 이어진다.

데이터셋: LiTS(Liver Tumor Segmentation, MICCAI 2017 챌린지) — Kaggle에
`andrewmvd/liver-tumor-segmentation`(Part 1) + `andrewmvd/liver-tumor-segmentation-part-2`
(Part 2), 두 개로 나뉘어 올라와 있음(용량 제한 때문에 분할). MSD Spleen처럼
단일 기관에서 깨끗하게 재검수한 데이터가 아니라, **여러 병원에서 모은 CT라
슬라이스 두께(0.45~6mm)와 스캐너 프로토콜이 제각각인 실전형 데이터** —
"정제된 데이터가 아니라 지저분한 데이터에서도 성능을 뽑아낸다"는 의도로
RSNA 대신/추가로 선택.

라벨: 0=배경, 1=간, 2=종양 (RSNA의 2-class(spleen)보다 한 단계 더 어려운
multi-class 문제).

모델: MONAI SegResNet — 이미 healthcare-main 전체가 MONAI 기반이라 툴체인이
그대로 이어짐.

실행 (Kaggle):
    1. Kaggle 노트북에서 "Add Input"으로 `andrewmvd/liver-tumor-segmentation`과
       `andrewmvd/liver-tumor-segmentation-part-2` 둘 다 추가
    2. !pip install monai
    3. 이 파일 내용을 셀에 붙여넣고 실행

주의: 캐글 데이터셋 페이지의 정확한 하위 폴더명을 직접 확인하지 못해서,
`volume*.nii*` / `segmentation*.nii*` 패턴으로 `/kaggle/input/` 아래를
재귀적으로 훑어 파일명 안의 번호로 (image, label) 쌍을 맞추는 방식으로
짰다 — 실제 폴더 구조가 조금 달라도 동작하게 하기 위함. Kaggle에서 실행
전에 "Data" 탭에서 실제 폴더 구조를 한 번 확인해보는 걸 권장.
"""
import os
import re
import glob

import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandShiftIntensityd, EnsureTyped, Activations, AsDiscrete,
)
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_ROOT = "/kaggle/input"
ROI_SIZE = (96, 96, 96)
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
VAL_FRACTION = 0.1  # 환자 단위 90/10 분할


def find_volume_label_pairs():
    """
    /kaggle/input 전체를 재귀 탐색해서 volume-N.nii(.gz) / segmentation-N.nii(.gz)를
    번호(N)로 매칭한다. 정확한 중간 경로(owner/slug 등)를 가정하지 않고
    /kaggle/input 바로 아래부터 전부 훑는다 — 실제로는
    /kaggle/input/datasets/andrewmvd/liver-tumor-segmentation(-part-2)처럼
    /kaggle/input/liver-tumor-segmentation보다 한 단계 더 깊었던 것으로 확인됨.
    """
    volumes, labels = {}, {}
    for path in glob.glob(os.path.join(INPUT_ROOT, "**", "volume*.nii*"), recursive=True):
        m = re.search(r"(\d+)", os.path.basename(path))
        if m:
            volumes[int(m.group(1))] = path
    for path in glob.glob(os.path.join(INPUT_ROOT, "**", "segmentation*.nii*"), recursive=True):
        m = re.search(r"(\d+)", os.path.basename(path))
        if m:
            labels[int(m.group(1))] = path

    common_ids = sorted(set(volumes) & set(labels))
    if not common_ids:
        raise FileNotFoundError(
            "volume/segmentation 쌍을 찾지 못했다. Kaggle 'Data' 탭에서 실제 폴더 "
            "구조를 확인하고 INPUT_ROOTS / 매칭 패턴을 조정할 것."
        )
    print(f"매칭된 환자 케이스 {len(common_ids)}건 발견")
    return [{"image": volumes[i], "label": labels[i]} for i in common_ids]


def get_transforms(train=True):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # 간 CT는 비장(-57~164)보다 넓은 연조직 윈도우(-200~200 HU)를 흔히 씀
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
    if train:
        base += [
            # 일부 환자는 크롭 후 Z축(depth)이 ROI_SIZE(96)보다 얇음(슬라이스 두께가
            # 커서 총 슬라이스 수가 적은 케이스) — RandCropByPosNegLabeld 전에
            # 부족한 만큼 패딩해서 크롭 실패를 방지
            SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE, mode="constant"),
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=ROI_SIZE,
                pos=1, neg=1, num_samples=2, image_key="image", image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
    base.append(EnsureTyped(keys=["image", "label"]))
    return Compose(base)


def main():
    print(f"Device: {DEVICE}")
    data_dicts = find_volume_label_pairs()

    n_val = max(1, int(len(data_dicts) * VAL_FRACTION))
    train_files, val_files = data_dicts[n_val:], data_dicts[:n_val]
    print(f"train {len(train_files)} / val {len(val_files)}")

    train_ds = Dataset(data=train_files, transform=get_transforms(train=True))
    val_ds = Dataset(data=val_files, transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    # out_channels=3: 배경/간/종양
    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        init_filters=16, in_channels=1, out_channels=3,
    ).to(DEVICE)

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])
    post_label = Compose([AsDiscrete(to_onehot=3)])

    best_dice = 0.0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1} train"):
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        model.eval()
        dice_metric.reset()
        # "경계를 얼마나 정확히 그렸는가"(Dice)와 별개로, "케이스(환자) 단위로
        # 종양의 존재 자체를 놓치지 않았는가"를 센다 — 확진이 아니라 의심(스크리닝)
        # 관점의 핵심 지표. TUMOR_LABEL=2(배경=0, 간=1, 종양=2).
        tumor_tp = tumor_fn = tumor_fp = tumor_tn = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch+1} val"):
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = sliding_window_inference(images, roi_size=ROI_SIZE, sw_batch_size=4, predictor=model)
                outputs = [post_pred(i) for i in decollate_batch(outputs)]
                labels_ = [post_label(i) for i in decollate_batch(labels)]
                dice_metric(y_pred=outputs, y=labels_)

                pred_classes = torch.argmax(outputs[0], dim=0)  # one-hot(3,D,H,W) -> (D,H,W)
                label_classes = torch.argmax(labels_[0], dim=0)
                gt_has_tumor = bool((label_classes == 2).any())
                pred_has_tumor = bool((pred_classes == 2).any())
                if gt_has_tumor and pred_has_tumor:
                    tumor_tp += 1
                elif gt_has_tumor and not pred_has_tumor:
                    tumor_fn += 1  # 놓침 — 이 케이스가 가장 치명적
                elif pred_has_tumor:
                    tumor_fp += 1
                else:
                    tumor_tn += 1
        val_dice = dice_metric.aggregate().item()  # 간+종양 평균 dice(경계 정확도)
        case_recall = tumor_tp / (tumor_tp + tumor_fn) if (tumor_tp + tumor_fn) > 0 else float("nan")
        case_fp_rate = tumor_fp / (tumor_fp + tumor_tn) if (tumor_fp + tumor_tn) > 0 else float("nan")

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] loss={epoch_loss:.4f} val_dice={val_dice:.4f} | "
            f"종양 놓침 없음(case recall)={case_recall:.3f} (TP={tumor_tp},FN={tumor_fn}) "
            f"오탐률(case FP rate)={case_fp_rate:.3f} (FP={tumor_fp},TN={tumor_tn})"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "ct_liver_seg_best.pth")
            print(f"  -> 최고 성능 갱신, ct_liver_seg_best.pth 저장 (dice={best_dice:.4f})")

    print(f"\n최종 최고 Dice(간+종양 평균): {best_dice:.4f}")


if __name__ == "__main__":
    main()
