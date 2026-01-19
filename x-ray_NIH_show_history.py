import pickle
import matplotlib.pyplot as plt

# 파일 경로
file_path = 'x-ray_NIH Chest X-rays_denseNet-121_v13.pkl'

# 'rb' (Read Binary) 모드로 읽어야 합니다.
with open(file_path, 'rb') as f:
    history = pickle.load(f)

def show_history(history):
    print(history.keys())

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if 'loss' in history and 'val_loss' in history:
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
    else:
        plt.title('Loss data missing')
    
    plt.subplot(1, 2, 2)
    if 'val_auc' in history and 'val_auc' in history:
        plt.plot(history['val_auc'], label='Validation auc')
        plt.title('Training and Validation Accuracy')
        plt.legend()
    else:
        plt.title('Accuracy data missing')

    plt.show()

# 그래프 출력
show_history(history)