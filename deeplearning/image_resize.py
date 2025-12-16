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
for img_name in tqdm(images):
    with Image.open(os.path.join(input_folder, img_name)) as img:
        # 224x224로 줄여서 저장
        img = img.resize((320, 320), Image.Resampling.BILINEAR)
        img.save(os.path.join(output_folder, img_name))
        
print("완료! 이제 학습 코드의 경로를 output_folder로 바꾸세요.")