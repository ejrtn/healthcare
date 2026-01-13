import pandas as pd
import os
import torch
import numpy as np
import pydicom
from scipy import ndimage
from monai.transforms import (
    Compose, LoadImaged, Spacingd, Orientationd, EnsureChannelFirstd,
    ScaleIntensityRanged, Resized, MapTransform, SelectItemsd, CopyItemsd, ConcatItemsd,
    DeleteItemsd
)
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed    # cpu 사용하기 해해

# 0. 설정 및 경로
BASE_DIR = '/kaggle/input/rsna-2023-abdominal-trauma-detection/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
CLASS_NAME_LIST = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury']


IMAGE_TARGET = (64,128,128)
NUM_SLICES = 64

# 전처리된 데이터를 저장할 폴더
SAVE_DIR = '/kaggle/working/'
os.makedirs(SAVE_DIR, exist_ok=True)

# 파일 읽기
train_df = pd.read_csv(f'{BASE_DIR}train_2024.csv') # 파일명 확인 필요 (보통 train.csv)
tags_df = pd.read_parquet(f'{BASE_DIR}train_dicom_tags.parquet')

# 고유 폴더 경로 추출 및 환자 ID 연결
tags_df['series_path'] = tags_df['path'].str.split('/').str[:-1].str.join('/')
unique_series = tags_df[['PatientID', 'series_path']].drop_duplicates()

data_dicts = []
for idx, row in unique_series.iterrows():
    p_id = int(row['PatientID'])
    s_path = row['series_path']
    
    # 해당 환자의 라벨 정보 가져오기
    patient_labels = train_df[train_df['patient_id'] == p_id]
    if len(patient_labels) == 0: continue # 라벨 없는 경우 제외
    labels = patient_labels.iloc[0]
    
    data_dicts.append({
        "image": f"{BASE_DIR}{s_path}",
        "patient_id": p_id,

        # 2진 분류 (Healthy, Injury) -> [1, 0] 또는 [0, 1] 형태가 됨
        "bowel": labels[['bowel_healthy', 'bowel_injury']].values.astype("float32"),
        "extravasation": labels[['extravasation_healthy', 'extravasation_injury']].values.astype("float32"),
        
        # 3중 분류 (Healthy, Low, High) -> [1, 0, 0], [0, 1, 0], [0, 0, 1] 형태가 됨
        "liver": labels[['liver_healthy', 'liver_low', 'liver_high']].values.astype("float32"),
        "kidney": labels[['kidney_healthy', 'kidney_low', 'kidney_high']].values.astype("float32"),
        "spleen": labels[['spleen_healthy', 'spleen_low', 'spleen_high']].values.astype("float32"),

        # any_injury가 1이면 "어딘가 이상함", 0이면 "완전 건강"
        "any_injury": np.array([1 - labels['any_injury'], labels['any_injury']]).astype("float32")
        
    })

patient_ids = train_df['patient_id'].unique()
train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
train_files = [d for d in data_dicts if d['patient_id'] in train_ids] # data_dicts에 patient_id 키 추가 필요
val_files = [d for d in data_dicts if d['patient_id'] in val_ids]

print(f"준비된 데이터 수: {len(data_dicts)}")
print(f"디바이스: {DEVICE}")

# ---------------------------------------------------------
# 방식 1: 직접 구현 고도화 (Manual)
# ---------------------------------------------------------
class CustomCTPreprocessor:
    def __init__(self, dicom_dir, target_shape, output_spacing=(1.5, 1.5, 1.5)):
        self.target_shape = target_shape
        self.output_spacing = output_spacing # 물리적 mm 단위 통일 (성능 향상의 핵심)

        self.result = self.process(dicom_dir)

    def load_and_sort_dicom(self, dicom_dir):
        """DICOM 로드 및 물리적 위치(Z축) 기준 정렬"""
        files = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in os.listdir(dicom_dir)]
        # ImagePositionPatient의 3번째 값(Z)으로 정렬해야 해부학적 순서가 맞음
        files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return files

    def get_hu_image(self, slices):
        """Raw Pixel을 물리적 밀도 단위(HU)로 변환"""
        image = np.stack([s.pixel_array for s in slices]).astype(np.float32)
        
        # 장비별 Rescale Slope/Intercept 적용
        slope = slices[0].RescaleSlope
        intercept = slices[0].RescaleIntercept
        image = image * slope + intercept
        return image

    def apply_multi_window(self, hu_image):
        """
        [성능 향상 팁] 3개의 서로 다른 윈도우를 RGB 채널처럼 사용
        - Channel 0: Soft Tissue (전반적 장기)
        - Channel 1: Liver/Spleen (고대비 장기 특화)
        - Channel 2: Bone/Air (경계선 강조)
        """
        def windowing(img, wl, ww):
            lower, upper = wl - ww//2, wl + ww//2
            img_clip = np.clip(img, lower, upper)
            return (img_clip - lower) / (upper - lower)

        # 임상적 근거 의한 값
        # ch0 (50, 400) - 복부 표준:
        # 복부 장기(간, 비장, 신장 등)의 평균 밀도가 보통 40~60 HU입니다. 그래서 중심을 50으로 잡습니다.
        # 주변의 지방(-50)부터 약간의 석회화(+200)까지 넓게 보기 위해 폭을 400으로 설정합니다.
        # ch1 (30, 150) - 간 특화(고대비):
        # 간 내부의 미세한 출혈이나 종양은 주변 조직과 밀도 차이가 아주 적습니다(약 10~20 HU 차이).
        # 이걸 잡아내려면 폭(WW)을 아주 좁게(150) 줄여서 대비를 극대화해야 합니다. 그래야 미세하게 어두운 부분이 확연히 드러납니다.
        # ch2 (100, 700) - 광범위/뼈/혈관:
        # 조영제가 들어간 혈관이나 뼈 근처의 출혈은 밀도가 높습니다.
        # 더 높은 수치(+100 이상)까지 포함하면서, 전체적인 윤곽을 잃지 않기 위해 범위를 아주 넓게(700) 잡은 것입니다.
        ch0 = windowing(hu_image, 50, 400)   # Standard Abdomen
        ch1 = windowing(hu_image, 30, 150)   # High Contrast Liver
        ch2 = windowing(hu_image, 100, 700)  # Wide Range (Bone/Fluid)
        
        return np.stack([ch0, ch1, ch2], axis=0) # (3, D, H, W)

    def resample_isotropic(self, image, slices):
        """
        병원마다 다른 슬라이스 두께를 1.5mm로 통일
        이 과정을 거쳐야 모델이 장기의 '진짜 크기'를 배움
        픽셀 1개가 실제 몸속에서 몇 mm인가? 맞추는 작업입니다. 
        """
        # 현재 Spacing (Thickness, PixelSpacing_X, PixelSpacing_Y)
        current_spacing = np.array([
            float(slices[0].SliceThickness),
            float(slices[0].PixelSpacing[0]),
            float(slices[0].PixelSpacing[1])
        ])
        
        resize_factor = current_spacing / self.output_spacing
        # ndimage.zoom으로 물리적 비율 보정 (채널별로 반복)
        new_channels = []
        for c in range(image.shape[0]):
            resampled = ndimage.zoom(image[c], resize_factor, order=1)
            new_channels.append(resampled)
        
        return np.stack(new_channels, axis=0)

    def final_resize_and_norm(self, image):
        """
        최종 크기 조정 및 Z-Score 정규화
        단순히 이미지를 가로, 세로, 높이 128개의 칸으로 강제로 늘리거나 줄이는 것
        """
        # 1. 모델 규격(64x128x128)으로 리사이즈
        factors = [
            1.0, # Channel은 고정
            self.target_shape[0] / image.shape[1],
            self.target_shape[1] / image.shape[2],
            self.target_shape[2] / image.shape[3]
        ]
        image = ndimage.zoom(image, factors, order=1)
        
        # 2. Z-Score 정규화: (x - mean) / std
        # 0~1 정규화보다 모델의 수렴 속도가 훨씬 빠름

        # 1e-8 더하는 유유
        # 만약 특정 슬라이스(image[c])의 모든 픽셀 값이 똑같다면(예: 전부 검은색 배경만 있는 경우),
        # 해당 이미지의 표준편차(std)는 0이되어 나눌 수 없게 됩니다.
        # 그래서 float32에서 1e-8 / float16 1e-5 정도가 가장 적당한 "아주 작은 수"를 더합니다.
        # 그냥 무작정 너무 작게 잡으면 0으로 인실 할 수 있음
        for c in range(image.shape[0]):
            image[c] = (image[c] - image[c].mean()) / (image[c].std() + 1e-8)
            
        return image
        
    def process(self, dicom_dir):
        """전체 파이프라인 실행"""
        slices = self.load_and_sort_dicom(dicom_dir)
        hu_img = self.get_hu_image(slices)
        multi_win = self.apply_multi_window(hu_img)
        resampled = self.resample_isotropic(multi_win, slices)
        final_img = self.final_resize_and_norm(resampled)
        return final_img.astype(np.float32) # 최종 출력: (3, 128, 128, 128)


# ---------------------------------------------------------
# 방식 2: MONAI (비교를 위해 단계를 Manual과 맞춤)
# ---------------------------------------------------------
def get_monai_expert_pipeline():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # 1. 물리적 해상도 통일 (Isotropic Resampling)
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        
        # 2. 멀티 윈도우 채널 생성 (이미지를 3개로 복사)
        CopyItemsd(keys=["image"], times=3, names=["img_soft", "img_liver", "img_bone"]),
        
        # 3. 각 복사본에 서로 다른 윈도우 적용
        ScaleIntensityRanged(keys=["img_soft"], a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["img_liver"], a_min=-50, a_max=100, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["img_bone"], a_min=-100, a_max=600, b_min=0.0, b_max=1.0, clip=True),
        
        # 4. 3개 채널을 하나로 합침 (3, 64, 128, 128)
        ConcatItemsd(keys=["img_soft", "img_liver", "img_bone"], name="image"),
        DeleteItemsd(keys=["img_soft", "img_liver", "img_bone"]),

        # 5. 최종 크기 조정 및 배경 제거 효과
        Resized(keys=["image"], spatial_size=IMAGE_TARGET)
    ])


def process_one_item(idx, item, total_count, mode, pipeline=None):

    step = max(1, total_count // 100)

    if idx % step == 0:
        percent = (idx / total_count) * 100
        # flush=True를 써야 백그라운드 로그에 즉시 기록됩니다.
        print(f"[{mode}] Progress: {percent:.0f}% 완료 ({idx}/{total_count})", flush=True)
        
    """
    한 명의 환자 데이터를 전처리하고 파일로 저장하는 핵심 함수
    """
    p_id = item['patient_id']
    s_id = item['image'].split('/')[-1]
    
    # 저장 경로 설정 (128 사이즈 구분을 위해 이름에 포함 가능)
    save_path = os.path.join(SAVE_DIR, f"{mode}_{p_id}_{s_id}.npz")
    
    # 1. 이미 파일이 있으면 전처리 생략하고 바로 리턴 (시간 절약)
    if os.path.exists(save_path):
        new_item = item.copy()
        new_item['image'] = save_path
        return new_item

    try:
        # 2. 방식에 따른 전처리 수행
        if mode == "manual":
            # Manual 방식: 직접 짠 CustomCTPreprocessor 호출
            # TARGET_SIZE는 전역 변수(예: (128, 128, 128))를 참조합니다.
            pre = CustomCTPreprocessor(item['image'], target_shape=IMAGE_TARGET)
            img = pre.result.astype(np.float16)
        else:
            # MONAI 방식: 전달받은 monai_pipeline 호출
            processed = pipeline(item)
            img = processed["image"].detach().cpu().numpy().astype(np.float16)
            
        # (Channel, Depth, H, W) -> (Slices, Channel, H, W)
        # 결과 형태: (64, 3, 128, 128)
        img = np.transpose(img, (1, 0, 2, 3))
        
        # 3. 압축률이 적어 의미 없다 판단하여 그냥 저장
        np.savez_compressed(save_path, img)
        
        # 4. 경로를 업데이트한 새 딕셔너리 반환
        new_item = item.copy()
        new_item['image'] = save_path
        return new_item

    except Exception as e:
        print(f"[{mode}] Error ID {p_id}: {e}")
        return None


    

class LoadNpyTransformd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        # npz 파일 로드
        with np.load(d["image"]) as data_file:
            # 저장할 때 사용했던 키인 'img'를 사용해 데이터를 꺼냅니다.
            img = data_file['img']
            
        # numpy 배열을 torch 텐서로 변환합니다.
        d["image"] = torch.from_numpy(img).float()
        return d

if __name__ == "__main__":
    # Manual 전처리
    print("Train 데이터 전처리 시작 (Parallel)...")
    # 1000개만 선택
    train_files_subset = train_files[2000:]
    total_len = len(train_files_subset) # 진행률 표시를 위해 1000으로 설정

    train_results = Parallel(n_jobs=-1)(
        delayed(process_one_item)(idx, item, total_len, "manual") 
        for idx, item in enumerate(train_files_subset)
    )
    # 에포크 에러 방지를 위해 None 제거 (주소록 업데이트)
    train_files_preprocessed = [r for r in train_results if r is not None]
    print("Train 데이터 전처리 종료 (Parallel)...")

    print("Val 데이터 전처리 시작 (Parallel)...")
    total_len = len(val_files)
    val_results = Parallel(n_jobs=-1)(
        delayed(process_one_item)(idx, item, total_len, "manual") for idx,item in enumerate(val_files)
    )
    # 에포크 에러 방지를 위해 None 제거 (주소록 업데이트)
    val_files_preprocessed = [r for r in val_results if r is not None]
    print("Val 데이터 전처리 종료 (Parallel)...")

    # MONAI 전리리
    pipeline = get_monai_expert_pipeline()

    # 1000개만 선택
    train_files_subset = train_files[:2000]
    total_len = len(train_files_subset) # 진행률 표시를 위해 1000으로 설정

    print("Train 데이터 전처리 시작 (Parallel)...")
    train_results = Parallel(n_jobs=-1)(
        delayed(process_one_item)(idx, item, total_len, "monai", pipeline)
        for idx,item in enumerate(train_files_subset)
    )
    # 에포크 에러 방지를 위해 None 제거 (주소록 업데이트)
    train_files_preprocessed = [r for r in train_results if r is not None]
    print("Train 데이터 전처리 종료 (Parallel)...")

    print("Val 데이터 전처리 시작 (Parallel)...")
    total_len = len(val_files) # 진행률 표시를 위해 1000으로 설정
    val_results = Parallel(n_jobs=-1)(
        delayed(process_one_item)(idx, item, total_len, "monai", pipeline)
        for idx,item in enumerate(val_files)
    )
    # 에포크 에러 방지를 위해 None 제거 (주소록 업데이트)
    val_files_preprocessed = [r for r in val_results if r is not None]
    print("Val 데이터 전처리 종료 (Parallel)...")
