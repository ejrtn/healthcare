"""
Segmentation 결과로 3D 복원(Reconstruction) — Kaggle/Colab용.

`ct_organ_segmentation.py`에서 학습한 모델로 간(liver)을 슬라이스마다
segmentation하면, 그 결과는 "2D 마스크가 여러 장 쌓인 것"일 뿐이다. 여기서는
그 마스크 볼륨에 **Marching Cubes 알고리즘**을 적용해서 실제 3D 표면(mesh)을
복원한다 — 딥러닝이 아니라 고전 알고리즘이지만, "여러 2D 단면에서 원래의 입체
형태를 재구성한다"는 3D reconstruction의 정의 그대로다.

파이프라인: CT 슬라이스 → (딥러닝) segmentation mask(배경/간/종양) → (고전
알고리즘) 간 표면 3D 복원 → 시각화. 앞 단계(segmentation) 결과를 그대로
이어받아 쓰기 때문에 새 데이터셋이 필요 없다.

실행 (Kaggle/Colab):
    !pip install monai scikit-image
    (ct_organ_segmentation.py를 먼저 실행해 ct_liver_seg_best.pth를 만든 뒤 실행)
"""
import torch
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from monai.data import DataLoader, Dataset
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped, Activations, AsDiscrete,
)

from ct_organ_segmentation import find_volume_label_pairs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ct_liver_seg_best.pth"
LIVER_CHANNEL = 1  # 0=배경, 1=간, 2=종양


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])


def load_model():
    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        init_filters=16, in_channels=1, out_channels=3,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict_volume_mask(model, image_tensor):
    """한 환자의 3D 볼륨 전체에 대해 segmentation을 돌려서 (배경/간/종양) 마스크 볼륨을 얻는다."""
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
    with torch.no_grad():
        from monai.inferers import sliding_window_inference
        outputs = sliding_window_inference(
            image_tensor.to(DEVICE), roi_size=(96, 96, 96), sw_batch_size=2, predictor=model,
        )
        mask = post_pred(outputs[0]).cpu().numpy()[0]  # (D, H, W), 0/1/2
    return mask


def reconstruct_3d_surface(mask_volume, target_label=LIVER_CHANNEL, spacing=(1.5, 1.5, 2.0)):
    """
    Marching Cubes로 지정한 라벨(기본: 간)의 이진 마스크 볼륨에서 3D 표면(정점,
    삼각형면)을 추출한다. spacing은 실제 복셀 간격(mm) — 앞서 Spacingd에서 맞춘
    값과 일치시켜야 실제 해부학적 비율이 맞는 3D 모델이 나온다.
    """
    binary_volume = (mask_volume == target_label).astype(np.float32)
    verts, faces, normals, values = measure.marching_cubes(
        binary_volume, level=0.5, spacing=spacing,
    )
    print(f"복원된 3D 표면(label={target_label}): 정점 {len(verts)}개, 삼각형면 {len(faces)}개")
    return verts, faces


def visualize_3d(verts, faces, save_path="liver_3d_reconstruction.png"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor([0.7, 0.3, 0.3])
    ax.add_collection3d(mesh)

    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_title("Liver 3D Reconstruction (Marching Cubes)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"3D 복원 결과를 {save_path}에 저장했습니다.")
    plt.show()


def main():
    print(f"Device: {DEVICE}")
    data_dicts = find_volume_label_pairs()
    n_val = max(1, int(len(data_dicts) * 0.1))
    val_files = data_dicts[:n_val]

    val_ds = Dataset(data=val_files, transform=get_val_transforms())
    loader = DataLoader(val_ds, batch_size=1)

    model = load_model()

    sample = next(iter(loader))
    mask_volume = predict_volume_mask(model, sample["image"])

    verts, faces = reconstruct_3d_surface(mask_volume)
    visualize_3d(verts, faces)


if __name__ == "__main__":
    main()
