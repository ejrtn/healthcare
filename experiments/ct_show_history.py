import pickle
import matplotlib.pyplot as plt

# 파일 경로
file_path = 'assets/monai_ct_convnext_v14_2.pkl'

# 'rb' (Read Binary) 모드로 읽어야 합니다.
with open(file_path, 'rb') as f:
    history = pickle.load(f)

def show_history(history):
    plt.figure(figsize=(15, 6))

    # 1. Loss 그래프 (Training vs Validation)
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val Loss", marker='o')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # 2. Mean AUC 그래프
    # 키 이름을 val_auc_mean으로 수정했습니다.
    plt.subplot(1, 3, 2)
    plt.plot(history["auc_avg_loss"], label="Mean Val AUC", color='orange', marker='s')
    plt.title("Mean Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.legend()
    
    # 장기별로 리스트를 추출하여 그래프 그리기
    plt.subplot(1, 3, 3) # 1행 3열 중 3번째 (에러 해결 지점)

    for organ in history["auc_details"][0].keys():
        # 각 장기별 데이터를 추출하여 루프 안에서 그립니다.
        organ_auc_history = [epoch_data[organ] for epoch_data in history["auc_details"]]
        plt.plot(organ_auc_history, label=f"{organ}")
    
    plt.title("Validation AUC by Organ")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.ylim(0.4, 1.05) # AUC가 1일 수도 있으므로 1.05 정도로 설정
    plt.grid(True, linestyle='--')
    # 범례가 많을 수 있으므로 그래프 옆으로 뺍니다.
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small') 
    
    plt.tight_layout()
    plt.show()

# 데이터 확인
print("불러온 히스토리 키:", history.keys())

# 그래프 출력
show_history(history)