import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
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
RESULTS_DIR = 'D:/healthcare/healthcare/static/results/'
NUM_CLASSES = 14
IMG_SIZE = 320
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# ------------------------------------------------------------------
# 2. 모델 로드 (전역 변수로 1회만 실행)
# ------------------------------------------------------------------
print("⏳ 모델 및 설정 로딩 중...")

model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, NUM_CLASSES)
)

try:
    # map_location: CPU 환경에서도 돌아가도록 안전장치
    state_dict = torch.load(os.path.join(current_dir, MODEL_PATH), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    model = None

# Grad-CAM 객체 미리 생성 (성능 최적화)
if model:
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

# ------------------------------------------------------------------
# 3. 분석 실행 함수
# ------------------------------------------------------------------
def run_analysis(image_path: str, uuid_file_name: str):
    if model is None:
        return {"error": "Model not loaded"}

    # 1. 이미지 불러오기
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return {"error": "Image file not found"}

    # -----------------------------------------------------------
    # [수정 1] 이미지 크기 통일 (Resize 이슈 해결)
    # 시각화용 이미지와 텐서 입력용 이미지의 크기가 같아야 에러가 안 납니다.
    # -----------------------------------------------------------

    # (B) 모델 입력용 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 2. 모델 예측 (Inference)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]

    # 3. 결과 데이터 준비
    response_data = {
        "filename": os.path.basename(image_path), # 전체 경로 대신 파일명만
        "predictions": {}
    }

    # 4. 라벨별 Grad-CAM 생성
    # (속도를 위해 상위 3개만 하거나, 필요시 전체 루프)
    for idx, label_name in enumerate(LABELS):
        
        # 해당 라벨의 확률
        prob_percent = probs[idx] * 100
        
        # Grad-CAM 타겟 설정
        targets = [ClassifierOutputTarget(idx)]
        
        # (A) Grayscale CAM 생성 (0.0 ~ 1.0)
        # input_tensor가 requires_grad=True 상태여야 함 (GradCAM 라이브러리가 알아서 처리)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # -------------------------------------------------------
        # [수정 2] 의료용 시각화 (Jet Colormap 적용 후 저장)
        # 프론트엔드에서 색칠하지 말고, 서버에서 예쁘게 칠해서 보냅니다.
        # -------------------------------------------------------
        
        # 1. 0~255 정수로 변환
        heatmap_uint8 = np.uint8(255 * grayscale_cam)
        
        # 2. Jet Colormap 적용 (파랑-초록-빨강)
        # OpenCV는 BGR을 쓰므로 나중에 RGB로 바꿔야 할 수도 있지만, 
        # PNG 저장 시 cv2.imwrite를 쓰면 BGR 그대로 저장해도 정상적으로 나옵니다.
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # 3. 히트맵 저장
        heatmap_filename = f"{uuid_file_name}_heatmap_{label_name}.png"
        heatmap_path = os.path.join(RESULTS_DIR, heatmap_filename)
        
        # OpenCV로 저장 (자동으로 BGR -> 파일 포맷에 맞게 저장됨)
        cv2.imwrite(heatmap_path, heatmap_color)
        
        # 4. 응답 데이터 추가
        # mask_url이라고 되어있지만, 실제로는 '컬러 히트맵' URL입니다.
        response_data["predictions"][label_name]={
            "probability": f"{prob_percent:.1f}%",
            "heatmap_url": heatmap_filename, # 프론트에서 이 파일명을 씁니다
            "raw_prob": float(probs[idx])    # 정렬용 숫자
        }

    return response_data