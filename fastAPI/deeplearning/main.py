import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import uuid

# Grad-CAM 라이브러리
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ------------------------------------------------------------------
# 1. 설정 (본인 경로에 맞게 수정 필수!)
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = 'x-ray_model_denseNet-121_v9.pth'
RESULTS_DIR = 'D:/healthcare/healthcare/data/images/'
NUM_CLASSES = 14
IMG_SIZE = 320
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
        model.load_state_dict(torch.load(os.path.join(current_dir, MODEL_PATH), map_location=device))
        print("✅ 모델 로드 성공!")
    except FileNotFoundError:
        print(f"❌ Error: 모델 파일({os.path.join(current_dir, MODEL_PATH)})을 찾을 수 없습니다.")
        return None
        
    model.to(device)
    model.eval()
    return model

# ------------------------------------------------------------------
# [핵심] 3. 직관적인 히트맵 생성 함수 (Clean Heatmap)
# ------------------------------------------------------------------
def visualize_cam_clean(model, input_tensor, original_img, target_category_index, threshold=0.2):

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
def run_analysis(image_path:str):
    # 1. 모델 준비
    model = load_model()
    if model is None: return

    # 2. 이미지 불러오기 및 전처리
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: 이미지 파일({image_path})을 찾을 수 없습니다.")
        return

    # 시각화용 이미지 (0~1 사이 실수형, Numpy 배열)
    vis_image = np.array(raw_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    # 모델 입력용 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 3. 예측 (Prediction)
    print("이미지 분석 중...")
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0] # 확률로 변환

    # 4. Grad-CAM 시각화 (히트맵 생성)
    target_layers = [model.features[-1]] # DenseNet의 마지막 특징 추출층
    cam = GradCAM(model=model, target_layers=target_layers)

    # 5. 결과 이미지 저장
    matplotlib.use('Agg')
    
    response_data = {
        "filename": Image.open(image_path).filename,
        "predictions": [] 
    }

    for idx, label_name in enumerate(LABELS):

        targets = [ClassifierOutputTarget(idx)]

        # CAM 생성
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # 원본 이미지 위에 덮어쓰기
        cam_image = show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)

        # (B) 그래프 그리기 (객체 지향 방식)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        
        # 왼쪽: 원본
        axes[0].imshow(vis_image) # vis_image는 원본 이미지 변수
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # 오른쪽: 해당 라벨의 Grad-CAM
        axes[1].imshow(cam_image) # 위에서 만든 cam_image 사용
        # 제목에 확률 표시
        prob_percent = probs[idx] * 100
        axes[1].set_title(f"Focus for: {label_name} ({prob_percent:.1f}%)")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # (C) 파일로 저장
        # 파일명 충돌 방지를 위해 UUID 사용
        unique_filename = f"{uuid.uuid4()}_{label_name}.png"
        save_path = os.path.join(RESULTS_DIR, unique_filename)
        
        fig.savefig(save_path)
        plt.close(fig) # 메모리 해제 (매우 중요!)
        
        # (D) 결과 리스트에 정보 추가
        # 웹에서 접근할 수 있는 URL 생성
        image_url = f"http://localhost:8000/results/{unique_filename}"
        
        response_data["predictions"].append({
            "label": label_name,
            "probability": f"{prob_percent:.1f}%",
            "image_url": image_url
        })
        
    # 3. JSON 데이터 반환 (이미지 파일 자체가 아님)
    return response_data