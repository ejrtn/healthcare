import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pickle  #파일저장에
#pip install tqdm
from tqdm import tqdm # 학습 진행 상황 시각화를 위해 추가
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import classification_report

import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 이후 PyTorch, NumPy, TensorFlow 등을 import 합니다.
# 경고 무시 설정
warnings.filterwarnings('ignore')

# 시드 고정 (재현성을 위해)
def set_seed(seed_value=42):
    random.seed(seed_value) # Python 기본 난수
    np.random.seed(seed_value) # NumPy 난수
    torch.manual_seed(seed_value) # PyTorch CPU 난수
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value) # PyTorch GPU 난수
        torch.cuda.manual_seed_all(seed_value) # PyTorch 모든 GPU 난수
        
        # CUDNN이 결정적 연산을 사용하도록 강제
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        
set_seed(42)

class NIHChestXrayDataset(Dataset):
    def __init__(self, metadata, img_dir, transform=None):
        self.metadata = metadata
        self.img_dir = img_dir
        self.transform = transform
        
        # 14가지 질병 클래스 정의 (No Finding 제외)
        self.all_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 이미지 경로 찾기
        img_name = self.metadata.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 이미지 로드 및 RGB 변환 (DenseNet은 3채널 입력 필요)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 레이블 처리 (Multi-label)
        label_str = self.metadata.iloc[idx]['Finding Labels']
        label_vec = torch.zeros(len(self.all_labels))
        
        for i, disease in enumerate(self.all_labels):
            if disease in label_str:
                label_vec[i] = 1.0
                
        return image, label_vec

# 1. 경로 설정 및 파라미터
TRAIN_VAL_LIST_CSV_PATH = './data/archive/Data_Entry_2017_train_val_list.csv'
TEST_LIST_CSV_PATH = './data/archive/Data_Entry_2017_test_list.csv'

TRAIN_VAL_IMG_DIR = './data/archive/images_resized_320/train_val_list' # 실제 경로 확인 필요
TEST_IMG_DIR = './data/archive/images_resized_320/test_list' # 실제 경로 확인 필요

# 데이터 CSV 로드
full_df = pd.read_csv(TRAIN_VAL_LIST_CSV_PATH)

# Train / Validation 분리 (8:2)
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

# --- 1. 설정 (Configuration) ---
BATCH_SIZE = 16  # 메모리에 맞춰 조절 (16 or 32)
IMG_SIZE = 320   # Pre-trained 모델 표준 크기
LEARNING_RATE = 0.0001 # 전이 학습은 학습률을 낮게 잡습니다
NUM_EPOCHS = 10
NUM_CLASSES = 14
OUTPUTS_OVER = 0.35

model_save_path_pth = 'x-ray_NIH Chest X-rays_denseNet-121_v13.pth'
history_filepath = 'x-ray_NIH Chest X-rays_denseNet-121_v13.pkl'

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")


# -----------------------------------------------------------------------------
# 3. 모델 정의 (DenseNet-121 Transfer Learning)
# -----------------------------------------------------------------------------
def get_model(num_classes):
    # ImageNet 가중치를 사용하는 DenseNet121 불러오기 (전이 학습의 핵심)
    # weights='DEFAULT'는 가장 최신 가중치를 가져옵니다.
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # DenseNet의 마지막 분류기 층(Classifier) 교체
    # 원래는 1000개를 분류하지만, 우리는 2개(정상/환자)만 분류하면 됨
    num_features = model.classifier.in_features # 원래 입력 특징 개수 (1024개)
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),                # 과적합 방지
        nn.Linear(num_features, num_classes)     # 최종 출력
    )
    return model

model = get_model(NUM_CLASSES)
model = model.to(device)


# 2. 데이터 로딩 및 학습 함수 (검증 단계 및 출력 추가)
def DataIncrease():

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),       # 좌우 반전
        transforms.RandomRotation(10),           # -10~10도 회전
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 명암 조절 중요
        transforms.ToTensor(),
        # ImageNet 학습 데이터의 평균/표준편차로 정규화 (필수)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #검증 데이터 전처리 (val_transforms)검증 단계에서는 데이터의 정확한 성능을 측정해야 하므로 무작위 변환을 피합니다.
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
                                                                            

    # DataLoader 생성
    train_dataset = NIHChestXrayDataset(train_df, TRAIN_VAL_IMG_DIR, transform=train_transforms)
    val_dataset = NIHChestXrayDataset(val_df, TRAIN_VAL_IMG_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # # BCEWithLogitsLoss 사용 (Sigmoid가 내장되어 있어 수치적으로 더 안정적)
    # custom_weights = [1, 1, 2, 2, 1, 1, 20, 1, 1, 1, 1, 1, 1, 5]
    # pos_weight = torch.FloatTensor(custom_weights).to(device)

    all_labels = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    # 각 질병별 양성(환자) 데이터 개수 카운트
    pos_counts = []
    for label in all_labels:
        # 해당 질병이 포함된 행의 개수
        count = train_df['Finding Labels'].str.contains(label).sum()
        pos_counts.append(max(count, 1)) # 0으로 나누기 방지

    pos_counts = torch.tensor(pos_counts).float().to(device)
    total_count = len(train_df)
    neg_counts = total_count - pos_counts

    # 2. [핵심] 반비례 가중치 계산 공식 (pos_weight = 음성 개수 / 양성 개수)
    # 데이터가 적을수록 이 값이 커집니다.
    # 개수에 피례해 계산된 가중치: tensor([  9.5276,  50.3494,   9.0376,   5.2732,  20.4566,  17.2781,  98.4526,0
    #        31.9929,  29.8463,  60.6925,  60.4188,  69.3445,  38.2176, 628.2636], device='cuda:0')
    pos_weight = neg_counts / pos_counts


    # 루트 씌운 개수에 피례해 계산된 가중치 결과: tensor([ 3.0867,  7.0957,  3.0063,  2.2964,  4.5229,  4.1567,  9.9223,  5.6562,
    #          5.4632,  7.7905,  7.7730,  8.3273,  6.1820, 25.0652], device='cuda:0')
    pos_weight = torch.log1p(pos_weight)

    print("자동 계산된 가중치:", pos_weight) 

    # 3. 손실 함수에 적용
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    
    # 1. 스케일러 정의 (학습 루프 시작 전)
    scaler = torch.amp.GradScaler('cuda')

    print("PyTorch 모델 학습 시작...")
    for epoch in range(NUM_EPOCHS):

        if epoch == 2:
            for param in model.parameters():
                param.requires_grad = True

            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * 0.1

        # 훈련 단계 (Training Step)
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        #훈련중인 부분을 보이게 하려고 한다. 
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float() # 그 다음 float로 변환
            optimizer.zero_grad()

            # [수정] autocast 적용
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # [수정] scaler로 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > OUTPUTS_OVER).float()
            total_train += labels.numel()
            correct_train += (predicted == labels).sum().item()
            
            train_loop.set_postfix(batch_loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        history['loss'].append(avg_train_loss)
        history['accuracy'].append(train_acc)
        
        # 검증 단계 (Validation Step)
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > OUTPUTS_OVER).float()
                total_val += labels.numel()      
                correct_val += (predicted == labels).sum().item()
                
                val_loop.set_postfix(batch_loss=loss.item())

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)

        # 학습 루프(Validation 단계 끝난 후)에 추가
        scheduler.step(avg_val_loss)
        
        # 에포크별 결과 출력
        print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}]")
        print(f"  Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    
    print("학습 완료!")
    
    # 모델 저장  - model.state_dict() :가중치
    torch.save(model.state_dict(), model_save_path_pth)
    print(f"모델 가중치가 '{model_save_path_pth}'에 저장되었습니다.")
    
    # 히스토리 저장
    with open(history_filepath, 'wb') as file:
        pickle.dump(history, file)
    print(f"학습 히스토리가 '{history_filepath}'에 저장되었습니다.")


def train_data_load():
    model = get_model(NUM_CLASSES)
    model = model.to(device)
    
    history = None
    
    if not os.path.exists(model_save_path_pth):
        print(f"저장된 모델 파일이 없습니다: {model_save_path_pth}")
        return False, None, None
        
    try:
        # map_location 추가 (GPU/CPU 호환성)
        model.load_state_dict(torch.load(model_save_path_pth, map_location=device))
        print(f"모델 로드 성공: '{model_save_path_pth}'")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return False, None, None
    
    if os.path.exists(history_filepath):
        try:
            with open(history_filepath, 'rb') as file:
                history = pickle.load(file)
                print(f"히스토리 로드 성공.")
        except Exception as e:
            print(f"히스토리 로드 오류: {e}")
    else:
        print("히스토리 파일이 없습니다.")
    
    return True, history, model
    
# 3. 모델 로드 및 평가 함수 (LoadModels)
def LoadModels():
    rf, history, model = train_data_load()

    if not rf:
        return rf

    # 그래프 그리기
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if 'accuracy' in history and 'val_accuracy' in history:
        plt.plot(history['accuracy'], label='Training Acc')
        plt.plot(history['val_accuracy'], label='Validation Acc')
        plt.title('Training and Validation Accuracy')
        plt.legend()
    else:
        plt.title('Accuracy data missing')

    plt.subplot(1, 2, 2)
    if 'loss' in history and 'val_loss' in history:
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
    else:
        plt.title('Loss data missing')
        
    plt.show()

    return True, model

# --- 평가 및 예측 함수 수정 ---
def PredictAndReport():
    rf, history, model = train_data_load()
    if not rf: return

    if not os.path.exists(TEST_LIST_CSV_PATH):
        print("Test CSV 파일이 없습니다.")
        return

    test_df = pd.read_csv(TEST_LIST_CSV_PATH)
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = NIHChestXrayDataset(test_df, TEST_IMG_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_preds = []
    all_labels = []

    print("테스트 데이터 예측 중...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > OUTPUTS_OVER).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 14개 클래스 이름
    target_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    print(f"\n--- Classification Report (Threshold: {OUTPUTS_OVER}) ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

# 메인 실행 부분
if __name__ == "__main__":
    while True:
        print("\n--- 메뉴 ---")
        print("1. 데이터 학습")
        print("2. 모델 로드 및 시각화")
        print("3. 예측하기")
        print("4. 종료")
        sel = input("선택: ")
        
        if sel == "1":
            DataIncrease()
        elif sel == "2":
            LoadModels()
        elif sel == "3":
            PredictAndReport()
        elif sel == "4":
            break
        else:
            print("잘못된 입력입니다.")
