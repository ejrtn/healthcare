

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
            img = d[key]  # (C, S, H, W)
            s = img.shape[1]
            indices = np.linspace(0, s - 1, self.num_slices).astype(int)
            d[key] = img[:, indices, :, :]
        return d


class InjuryPreprocessor:
    """부상 여부 분류용 전처리 — 전신 컨텍스트를 넓게 보되(margin=20) crop은 완화한다."""

    def __init__(self, target_slices=64, target_size=224):
        self.transforms = Compose([
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),

            # 세 개의 HU 윈도우(wide/soft/bone)를 채널로 합쳐 하나의 이미지로 만든다.
            CopyItemsd(keys=["image"], times=3, names=["img1", "img2", "img3"]),
            ScaleIntensityRanged(keys=["img1"], a_min=-300, a_max=500, b_min=0, b_max=1, clip=True),   # wide
            ScaleIntensityRanged(keys=["img2"], a_min=-160, a_max=240, b_min=0, b_max=1, clip=True),   # soft tissue
            ScaleIntensityRanged(keys=["img3"], a_min=0, a_max=1000, b_min=0, b_max=1, clip=True),     # bone
            ConcatItemsd(keys=["img1", "img2", "img3"], name="image"),
            DeleteItemsd(keys=["img1", "img2", "img3"]),

            CropForegroundd(keys=["image"], source_key="image", margin=20),
            SelectiveSamplingd(keys=["image"], num_slices=target_slices),
            Resized(keys=["image"], spatial_size=(-1, target_size, target_size))
        ])


class DiagnosisPreprocessor:
    """병명 분류용 전처리 — 병변 국소 부위에 집중하도록 crop을 공격적으로(margin=2) 좁힌다."""

    def __init__(self, target_slices=64, target_size=224):
        self.transforms = Compose([
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.2, 1.2, 1.2), mode="bilinear"),

            # soft tissue / angio(혈관) / bowel(장) 세 윈도우를 채널로 합친다.
            CopyItemsd(keys=["image"], times=3, names=["img_soft", "img_angio", "img_bowel"]),
            ScaleIntensityRanged(keys=["img_soft"], a_min=-160, a_max=240, b_min=0, b_max=1, clip=True),
            ScaleIntensityRanged(keys=["img_angio"], a_min=-250, a_max=450, b_min=0, b_max=1, clip=True),
            ScaleIntensityRanged(keys=["img_bowel"], a_min=-300, a_max=200, b_min=0, b_max=1, clip=True),
            ConcatItemsd(keys=["img_soft", "img_angio", "img_bowel"], name="image"),
            DeleteItemsd(keys=["img_soft", "img_angio", "img_bowel"]),

            CropForegroundd(keys=["image"], source_key="image", margin=2),
            SelectiveSamplingd(keys=["image"], num_slices=target_slices),
            Resized(keys=["image"], spatial_size=(-1, target_size, target_size))
        ])