import pickle
import matplotlib.pyplot as plt

# 파일 경로
file_path = 'monai_ct_convnext_v5.pkl'

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

    plt.figure(figsize=(20, 7)) # 가로 길이를 조금 더 늘림
    epochs = range(len(history["train_loss"]))

    # 1. Loss 그래프
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o', color='tab:blue')
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker='o', color='tab:orange')
    
    # Loss 수치 표시 (가독성을 위해 마지막 점과 최고/최저 위주로 표시하거나 전체 표시)
    for i, v in enumerate(history["val_loss"]):
        plt.text(i, v + 0.02, f"{v:.3f}", fontsize=9, ha='center', color='tab:orange')

    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 2. Mean AUC 그래프
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["auc_avg_loss"], label="Mean Val AUC", color='orange', marker='s')
    
    # AUC 수치 표시
    for i, v in enumerate(history["auc_avg_loss"]):
        plt.text(i, v + 0.005, f"{v:.3f}", fontsize=10, ha='center', fontweight='bold')

    plt.title("Mean Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 3. 장기별 AUC 그래프
    plt.subplot(1, 3, 3)
    for organ in history["auc_details"][0].keys():
        organ_auc_history = [epoch_data[organ] for epoch_data in history["auc_details"]]
        line, = plt.plot(epochs, organ_auc_history, label=f"{organ}", marker='.')
        
        # 장기별 그래프는 선이 많아 마지막 수치만 표시 (안 그러면 겹쳐서 안 보임)
        last_val = organ_auc_history[-1]
        plt.text(len(epochs)-1, last_val, f"{last_val:.2f}", fontsize=9, color=line.get_color(), fontweight='bold')

    plt.title("Validation AUC by Organ")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.ylim(0.4, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

# 데이터 확인
print("불러온 히스토리 키:", history.keys())

# 그래프 출력
show_history(history)