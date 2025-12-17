import os
from PIL import Image
import shutil

# 원본 폴더와 저장할 폴더
# image_size = 320
# input_folder = 'D:/healthcare/healthcare/data/test_images'
# output_folder = f'D:/healthcare/healthcare/data/test_images_resized_{image_size}/' # 새로 만들 폴더

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# images = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# print("이미지 리사이징 시작...")
# for img_name in images:
#     with Image.open(os.path.join(input_folder, img_name)) as img:
#         # 줄여서 저장
#         img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
#         img.convert("RGB").save(os.path.join(output_folder, img_name))
        
# print("완료! 이제 학습 코드의 경로를 output_folder로 바꾸세요.")


class Image_resize:
    def __init__(self, image_path:str, image_size:int):
        self.image_path = image_path
        self.image_size = image_size

    def __call__(self):

        try:
            with Image.open(self.image_path) as img:
                resized_img = img.resize(self.image_size, Image.Resampling.LANCZOS)

                save_path = f'{self.image_path}_resized_{self.image_size}'

                resized_img.convert("RGB").save(save_path)

                return save_path
            
        except Exception as e:
            print(f"에러 발생: {e}")
            return None
        