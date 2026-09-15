"""학습 중 저장된 history pickle(loss/AUC 추이)을 그래프로 확인하는 스크립트."""
import pickle
import matplotlib.pyplot as plt

file_path = 'assets/monai_ct_convnext_v17.pkl'

with open(file_path, 'rb') as f:
    history = pickle.load(f)


def show_history(history):
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val Loss", marker='o')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["auc_avg_loss"], label="Mean Val AUC", color='orange', marker='s')
    plt.title("Mean Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    for organ in history["auc_details"][0].keys():
        organ_auc_history = [epoch_data[organ] for epoch_data in history["auc_details"]]
        plt.plot(organ_auc_history, label=f"{organ}")
    plt.title("Validation AUC by Organ")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.ylim(0.4, 1.05)
    plt.grid(True, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()


print("불러온 히스토리 키:", history.keys())
show_history(history)