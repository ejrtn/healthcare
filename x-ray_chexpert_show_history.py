import pickle
import matplotlib.pyplot as plt

# 파일 경로
file_path = 'x-ray_chexpert_denseNet-121_v1.pkl'

# 'rb' (Read Binary) 모드로 읽어야 합니다.
with open(file_path, 'rb') as f:
    history = pickle.load(f)

def show_history(history):
    plt.figure(figsize=(12, 5))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linestyle='--')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Loss Over Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Mean AUC와 Top 5 AUC
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='Mean AUC (All)', color='green', marker='o')
    plt.plot(history['val_auc_top5'], label='Top 5 AUC (Critical)', color='orange', marker='s')
    plt.axhline(y=0.5, color='gray', linestyle=':', label='Random Guess') # 0.5 기준선
    plt.title('AUC-ROC: All vs Top 5', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('AUC Score')
    plt.ylim(0.4, 1.0) # 의료 AI 점수대는 보통 이 범위 내에 있음
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout() # 그래프 간격 자동 조절
    
    plt.show()

# 그래프 출력
show_history(history)