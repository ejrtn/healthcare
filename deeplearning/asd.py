import os
import shutil
import pandas as pd

# with open('./data/archive/train_val_list.txt', 'r', encoding='utf-8') as f:
    
    # train_val_list.txt 파일 기준 이미지 정리
    # if not os.path.exists('data/archive/images/train_val_list/'):
    #     os.makedirs('data/archive/images/train_val_list/')
    #     print(f"폴더 생성 완료: train_val_list")
    # for line in f:
    #     path = 'data/archive/images/'+line.strip()
    #     move_path = 'data/archive/images/train_val_list/'+line.strip()
    #     shutil.move(path, move_path)

    # Data_Entry_2017 나누기
    # cs = open('./data/archive/Data_Entry_2017_train_val_list.csv', 'w', encoding='utf-8')
    # cs.write('Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]\n')
    # cs.close()
    
#     # batch = 1000
#     # lists = []
#     # c = 0
#     # df = pd.read_csv('./data/archive/Data_Entry_2017.csv')
#     # for line in f:
#     #     pick_df = df[df['Image Index']==line.strip()].copy()
#     #     lists.append(",".join(map(str, pick_df.values[0])) + '\n')
#     #     if batch <= c:
#     #         cs = open('./data/archive/Data_Entry_2017_train_val_list.csv', 'a', encoding='utf-8')
#     #         cs.writelines(lists)
#     #         cs.close()
#     #         lists.clear()
#     #         c = 0
#     #     c += 1
#     # cs = open('./data/archive/Data_Entry_2017_train_val_list.csv', 'a', encoding='utf-8')
#     # cs.writelines(lists)
#     # cs.close()
#     pass

# with open('./data/archive/test_list.txt', 'r', encoding='utf-8') as f:

#     # test_list.txt기준 이미지 정리
#     if not os.path.exists('data/archive/images/test_list/'):
#         os.makedirs('data/archive/images/test_list/')
#         print(f"폴더 생성 완료: test_list")
#     for line in f:
#         path = 'data/archive/images/'+line.strip()
#         move_path = 'data/archive/images/test_list/'+line.strip()
#         shutil.move(path, move_path)

#     # Data_Entry_2017 나누기
#     cs = open('./data/archive/Data_Entry_2017_test_list.csv', 'w', encoding='utf-8')
#     cs.write('Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]\n')
#     cs.close()

#     batch = 1000
#     lists = []
#     c = 0
#     df = pd.read_csv('./data/archive/Data_Entry_2017.csv')
#     for line in f:
#         pick_df = df[df['Image Index']==line.strip()].copy()
#         lists.append(",".join(map(str, pick_df.values[0])) + '\n')
#         if batch <= c:
#             cs = open('./data/archive/Data_Entry_2017_test_list.csv', 'a', encoding='utf-8')
#             cs.writelines(lists)
#             cs.close()
#             lists.clear()
#             c = 0
#         c += 1
#     cs = open('./data/archive/Data_Entry_2017_test_list.csv', 'a', encoding='utf-8')
#     cs.writelines(lists)
#     cs.close()
        





import os
from PIL import Image
from tqdm import tqdm

# 원본 폴더와 저장할 폴더
input_folder = './data/archive/images/test_list' 
output_folder = './data/archive/images_resized_320/test_list/' # 새로 만들 폴더

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

images = [f for f in os.listdir(input_folder) if f.endswith('.png')]

print("이미지 리사이징 시작...")
for img_name in tqdm(["00000032_042.png"]):
    with Image.open(os.path.join(input_folder, img_name)) as img:
        # 224x224로 줄여서 저장
        img = img.resize((320, 320), Image.Resampling.BILINEAR)
        img.save(os.path.join(output_folder, img_name))
        
print("완료! 이제 학습 코드의 경로를 output_folder로 바꾸세요.")


# from sklearn.model_selection import train_test_split
# import pandas as pd
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN_VAL_LIST_CSV_PATH = './data/archive/Data_Entry_2017_train_val_list.csv'

# # 데이터 CSV 로드
# full_df = pd.read_csv(TRAIN_VAL_LIST_CSV_PATH)

# # Train / Validation 분리 (8:2)
# train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)


# all_labels = [
#         'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
#         'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
#         'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
#     ]

# # 각 질병별 양성(환자) 데이터 개수 카운트
# pos_counts = []
# for label in all_labels:
#     # 해당 질병이 포함된 행의 개수
#     count = train_df['Finding Labels'].str.contains(label).sum()
#     pos_counts.append(max(count, 1)) # 0으로 나누기 방지
# pos_counts = torch.tensor(pos_counts).float().to(device)
# total_count = len(train_df)
# neg_counts = total_count - pos_counts
# # 2. [핵심] 반비례 가중치 계산 공식 (pos_weight = 음성 개수 / 양성 개수)
# # 데이터가 적을수록 이 값이 커집니다.
# # 개수에 피례해 계산된 가중치: tensor([  9.5276,  50.3494,   9.0376,   5.2732,  20.4566,  17.2781,  98.4526,
# #        31.9929,  29.8463,  60.6925,  60.4188,  69.3445,  38.2176, 628.2636], device='cuda:0')
# pos_weight = neg_counts / pos_counts
# pos_weight[6] = pos_weight[6] * 5.0

# # 루트 씌운 개수에 피례해 계산된 가중치 결과: tensor([ 3.0867,  7.0957,  3.0063,  2.2964,  4.5229,  4.1567,  9.9223,  5.6562,
# #          5.4632,  7.7905,  7.7730,  8.3273,  6.1820, 25.0652], device='cuda:0')
# pos_weight = torch.sqrt(pos_weight)
# print("자동 계산된 가중치:", pos_weight) 