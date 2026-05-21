

from monai.transforms import (
Compose, LoadImaged, Spacingd, Orientationd, EnsureChannelFirstd,
ScaleIntensityRanged, Resized, CopyItemsd, ConcatItemsd, DeleteItemsd, MapTransform,
CropForegroundd
)
import torch

class SelectiveSamplingd(MapTransform):
    def __init__(self, keys, num_slices=64):
        super().__init__(keys)
        self.num_slices = num_slices

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key] # (C, S, H, W)
            s = img.shape[1]
            # 64장을 뽑을 인덱스 계산
            indices = np.linspace(0, s - 1, self.num_slices).astype(int)
            d[key] = img[:, indices, :, :]
        return d

# 부상 여부
class InjuryPreprocessor:
    def __init__(self, target_slices=64, target_size=224):
        self.transforms = Compose([

            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),

            # 🔥 더 넓은 window (global context)
            CopyItemsd(keys=["image"], times=3, names=["img1", "img2", "img3"]),

            # wide
            ScaleIntensityRanged(keys=["img1"], a_min=-300, a_max=500, b_min=0, b_max=1, clip=True),
            # soft
            ScaleIntensityRanged(keys=["img2"], a_min=-160, a_max=240, b_min=0, b_max=1, clip=True),
            # bone-ish
            ScaleIntensityRanged(keys=["img3"], a_min=0, a_max=1000, b_min=0, b_max=1, clip=True),

            ConcatItemsd(keys=["img1", "img2", "img3"], name="image"),
            DeleteItemsd(keys=["img1", "img2", "img3"]),

            # 🔥 crop 완화 (context 유지)
            CropForegroundd(keys=["image"], source_key="image", margin=20),

            SelectiveSamplingd(keys=["image"], num_slices=target_slices),

            Resized(keys=["image"], spatial_size=(-1, target_size, target_size))
        ])


# 병명분류
class DiagnosisPreprocessor:
    def __init__(self, target_slices=64, target_size=224):
        self.transforms = Compose([

            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.2, 1.2, 1.2), mode="bilinear"),

            # 🔥 너 기존 방식 유지 (좋음)
            CopyItemsd(keys=["image"], times=3, names=["img_soft", "img_angio", "img_bowel"]),

            ScaleIntensityRanged(keys=["img_soft"], a_min=-160, a_max=240, b_min=0, b_max=1, clip=True),
            ScaleIntensityRanged(keys=["img_angio"], a_min=-250, a_max=450, b_min=0, b_max=1, clip=True),
            ScaleIntensityRanged(keys=["img_bowel"], a_min=-300, a_max=200, b_min=0, b_max=1, clip=True),

            ConcatItemsd(keys=["img_soft", "img_angio", "img_bowel"], name="image"),
            DeleteItemsd(keys=["img_soft", "img_angio", "img_bowel"]),

            # 🔥 aggressive crop (lesion focus)
            CropForegroundd(keys=["image"], source_key="image", margin=2),

            SelectiveSamplingd(keys=["image"], num_slices=target_slices),

            Resized(keys=["image"], spatial_size=(-1, target_size, target_size))
        ])