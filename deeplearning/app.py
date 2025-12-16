import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Grad-CAM 라이브러리
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ------------------------------------------------------------------
# 1. 설정 (본인 경로에 맞게 수정 필수!)
# ------------------------------------------------------------------
MODEL_PATH = 'x-ray_model_denseNet-121_v4.pth'
TEST_IMAGE_PATH = './data/archive/images_resized_224/test_list/00000041_002.png'

NUM_CLASSES = 14
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# ------------------------------------------------------------------
# 2. 모델 로드 함수
# ------------------------------------------------------------------
def load_model():
    print("모델 로딩 중...")
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, NUM_CLASSES)
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ 모델 로드 성공!")
    except FileNotFoundError:
        print(f"❌ Error: 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
        return None
        
    model.to(device)
    model.eval()
    return model

# ------------------------------------------------------------------
# [핵심] 3. 직관적인 히트맵 생성 함수 (Clean Heatmap)
# ------------------------------------------------------------------
def visualize_cam_clean(model, input_tensor, original_img, target_category_index, threshold=0.2):
    """
    파란색 배경을 없애고, 중요한 부분만 붉게 표시하는 함수
    threshold: 이 값보다 낮은 중요도는 투명하게 처리 (0.0 ~ 1.0)
    """
    # 1. Grad-CAM 객체 생성
    target_layers = [model.features[-1]] # DenseNet 마지막 층
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_category_index)]

    # 2. 히트맵 추출 (0~1 사이 값)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 3. 노이즈 제거 (Thresholding)
    # 중요도가 낮은 부분(배경)은 0으로 만듦
    grayscale_cam[grayscale_cam < threshold] = 0

    # 4. 컬러맵 적용 (JET: 파랑~빨강, 하지만 파랑은 아래에서 제거됨)
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    heatmap = heatmap[..., ::-1] # OpenCV BGR -> RGB 변환

    # 5. 원본과 합성 (Alpha Blending)
    # 히트맵 값이 있는 부분만 색을 입히고, 나머지는 원본 그대로 둠
    cam_image = original_img.copy()
    
    # 히트맵의 강도(grayscale_cam)를 투명도(Alpha)로 사용
    # 강한 부분은 빨갛게, 약한 부분은 원본 그대로
    for c in range(3):
        cam_image[:, :, c] = original_img[:, :, c] * (1 - grayscale_cam) + heatmap[:, :, c] * grayscale_cam

    # 값 범위 안전장치 (0~1)
    cam_image = np.clip(cam_image, 0, 1)
    
    return cam_image

# ------------------------------------------------------------------
# 4. 메인 실행 로직
# ------------------------------------------------------------------
def run_analysis():
    # 1. 모델 준비
    model = load_model()
    if model is None: return

    # 2. 이미지 준비
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: 이미지 파일({TEST_IMAGE_PATH})이 없습니다.")
        return

    raw_image = Image.open(TEST_IMAGE_PATH).convert('RGB')
    
    # 시각화용 이미지 (0~1 실수형)
    vis_image = np.array(raw_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    
    # 모델 입력용
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 3. 예측
    print("🔍 이미지 분석 중...")
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]

    # 4. 결과 텍스트
    top3_indices = probs.argsort()[-3:][::-1]
    
    print("\n" + "="*40)
    print(f"🧪 분석 결과 (파일: {os.path.basename(TEST_IMAGE_PATH)})")
    print("="*40)
    for idx in top3_indices:
        print(f" -> {LABELS[idx]}: {probs[idx]*100:.2f}%")
    print("="*40)

    # 5. [수정됨] 직관적인 히트맵 생성
    highest_idx = top3_indices[0]
    
    # threshold=0.2 : 하위 20%의 약한 신호는 지워서 배경을 깨끗하게 만듦
    cam_image = visualize_cam_clean(model, input_tensor, vis_image, highest_idx, threshold=0.2)

    # 6. 화면 출력
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(vis_image)
    plt.title("Original X-ray")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"AI Focus: {LABELS[highest_idx]} ({probs[highest_idx]*100:.1f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 실행
if __name__ == "__main__":
    run_analysis()