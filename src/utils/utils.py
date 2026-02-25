import matplotlib.pyplot as plt
import numpy as np

def plot_ct_history(history, class_names):
    """
    Visualize training and validation loss, and organ-wise AUC.
    """
    plt.figure(figsize=(15, 6))

    # 1. Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val Loss", marker='o')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # 2. Mean AUC Plot
    plt.subplot(1, 3, 2)
    plt.plot(history["auc_avg_loss"], label="Mean Val AUC", color='orange', marker='s')
    plt.title("Mean Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.legend()

    # 3. Organ-wise AUC Plot
    plt.subplot(1, 3, 3)
    for organ in class_names:
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

def plot_xray_results(nih_history, chex_history, common_diseases):
    """
    Compare NIH and CheXpert dataset performance.
    """
    epochs = range(1, len(nih_history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Validation Loss Comparison
    plt.subplot(2, 2, 1)
    plt.plot(epochs, nih_history['val_loss'], 'r--', label='NIH Val Loss')
    plt.plot(epochs, chex_history['val_loss'], 'b-', label='CheXpert Val Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()

    # Mean AUC Comparison
    plt.subplot(2, 2, 2)
    plt.plot(epochs, nih_history['val_auc'], 'r--', label='NIH Mean AUC')
    plt.plot(epochs, chex_history['val_auc'], 'b-', label='CheXpert Mean AUC')
    plt.title('Overall Mean AUC comparison')
    plt.legend()

    # Disease Specific Comparisons (First 2 common diseases)
    for i, disease in enumerate(common_diseases[:2]):
        plt.subplot(2, 2, 3 + i)
        plt.plot(epochs, nih_history[f'auc_{disease}'], 'r--', label=f'NIH {disease}')
        plt.plot(epochs, chex_history[f'auc_{disease}'], 'b-', label=f'CheXpert {disease}')
        plt.title(f'{disease} AUC Comparison')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
