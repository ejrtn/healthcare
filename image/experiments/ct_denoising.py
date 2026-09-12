"""
저선량 CT Denoising (Image-to-image translation) — Kaggle/Colab GPU용.

지금까지의 태스크(분류, segmentation)는 "이미지 → 라벨/마스크"였는데, 이건
"이미지 → 다른 이미지"를 만들어내는 태스크다. 실제 병원에서 저선량 CT(방사선
피폭을 줄이려고 약하게 찍은 CT, 노이즈가 심함)를 고선량 CT처럼 깨끗하게
복원해주는 데 쓰이는 기술과 같은 원리.

정답이 있는 저선량/고선량 CT 쌍 데이터셋(AAPM-Mayo Clinic Low Dose CT Grand
Challenge)은 별도 신청 절차가 있어서 접근이 번거롭다. 대신 여기서는 **자기지도
학습(self-supervised)** 방식을 쓴다: 이미 갖고 있는 CT(LiTS, `ct_organ_
segmentation.py`와 같은 데이터 — 여러 병원에서 모은 CT라 원래도 스캐너별로
화질 편차가 있는 실전형 데이터)에 우리가 직접 저선량 CT와 비슷한 노이즈
(포아송+가우시안)를 인위적으로 씌우고, 그걸 원래 이미지로 복원하도록
학습시킨다 — 새 데이터셋이 필요 없고, 같은 CT 데이터를 계속 재사용하는
셈이라 healthcare-main 전체가 하나의 데이터로 일관되게 이어진다.

모델: 2D 슬라이스 단위 Denoising U-Net (MONAI의 UNet, out_channels=1로 이미지
자체를 출력 — segmentation과 구조는 거의 같고 마지막 출력만 다름)

실행 (Kaggle/Colab GPU):
    !pip install monai
    (Kaggle "Add Input"으로 andrewmvd/liver-tumor-segmentation +
    liver-tumor-segmentation-part-2 둘 다 추가한 상태라고 가정)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.networks.nets import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt

from ct_organ_segmentation import find_volume_label_pairs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
NOISE_STD = 0.15  # 저선량 CT를 흉내내는 노이즈 강도


def add_synthetic_low_dose_noise(image, noise_std=NOISE_STD):
    """
    저선량 CT는 광자 수가 적어 포아송 노이즈가 도드라지고, 여기에 검출기
    자체의 가우시안 노이즈도 섞인다 — 실제 저선량 촬영의 노이즈 특성을
    단순화해서 흉내낸 것.
    """
    poisson_noise = np.random.poisson(image * 50) / 50.0 - image
    gaussian_noise = np.random.normal(0, noise_std, image.shape)
    noisy = image + poisson_noise * 0.5 + gaussian_noise
    return np.clip(noisy, 0, 1)


class CTDenoiseDataset(Dataset):
    """CT 볼륨에서 2D 슬라이스를 하나씩 뽑아 (노이즈 낀 버전, 원본) 쌍을 만든다."""

    def __init__(self, nii_paths, max_slices_per_vol=30):
        import cv2
        self.slices = []
        for path in nii_paths:
            vol = nib.load(path).get_fdata()
            # 간 CT는 -200~200 HU 연조직 윈도우 사용 (ct_organ_segmentation.py와 동일)
            vol = np.clip(vol, -200, 200)
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
            valid_indices = [i for i in range(vol.shape[2]) if vol[:, :, i].std() > 0.01]
            if len(valid_indices) > max_slices_per_vol:
                step = len(valid_indices) // max_slices_per_vol
                valid_indices = valid_indices[::step][:max_slices_per_vol]
            for i in valid_indices:
                sl = vol[:, :, i]
                sl_128 = cv2.resize(sl.astype(np.float32), (128, 128))
                self.slices.append(sl_128)
        print(f"총 {len(self.slices)}개 슬라이스 로드 (메모리 최적화 완료)")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        clean = self.slices[idx]
        noisy = add_synthetic_low_dose_noise(clean)
        return (
            torch.tensor(noisy).unsqueeze(0).float(),
            torch.tensor(clean).unsqueeze(0).float(),
        )


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def main():
    print(f"Device: {DEVICE}")
    data_dicts = find_volume_label_pairs()
    nii_paths = [d["image"] for d in data_dicts]
    print(f"CT 볼륨 {len(nii_paths)}개 발견")

    n_val = max(1, len(nii_paths) // 10)
    train_paths, val_paths = nii_paths[n_val:], nii_paths[:n_val]

    train_ds = CTDenoiseDataset(train_paths)
    val_ds = CTDenoiseDataset(val_paths)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)

    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128), strides=(2, 2, 2),
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_psnr = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for noisy, clean in tqdm(train_loader, desc=f"epoch {epoch+1} train"):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader, desc=f"epoch {epoch+1} val"):
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                output = model(noisy).clamp(0, 1)
                val_psnr += psnr(output, clean)
        val_psnr /= len(val_loader)

        print(f"[Epoch {epoch+1}/{EPOCHS}] train_loss={train_loss:.4f} val_PSNR={val_psnr:.2f}dB")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "ct_denoise_best.pth")
            print(f"  -> 최고 성능 갱신, ct_denoise_best.pth 저장 (PSNR={best_psnr:.2f}dB)")

    # 노이즈/원본/복원 결과를 나란히 시각화해서 저장
    model.eval()
    noisy, clean = next(iter(val_loader))
    with torch.no_grad():
        denoised = model(noisy.to(DEVICE)).clamp(0, 1).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(noisy[0, 0], cmap="gray"); axes[0].set_title("저선량(노이즈)")
    axes[1].imshow(denoised[0, 0], cmap="gray"); axes[1].set_title("AI 복원 결과")
    axes[2].imshow(clean[0, 0], cmap="gray"); axes[2].set_title("원본(정답)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("ct_denoise_comparison.png", dpi=150)
    print(f"\n최종 최고 PSNR: {best_psnr:.2f}dB")
    print("ct_denoise_comparison.png 저장 완료")


if __name__ == "__main__":
    main()
