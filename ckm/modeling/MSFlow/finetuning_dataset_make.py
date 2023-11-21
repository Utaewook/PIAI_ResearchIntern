import random
import os
import shutil
from PIL import Image

src_filepath = 'E:\\codes\\ckm\\dataset\\CKM\\train\\good'
dest_filepath = 'E:\\codes\\ckm\\dataset\\CKM_finetuning\\train\\good'

# 이미지를 저장할 디렉토리 생성
os.makedirs(dest_filepath, exist_ok=True)

# 하위 디렉토리의 모든 이미지 파일 가져오기
image_files = [os.path.join(src_filepath, filename) for filename in os.listdir(src_filepath) if filename.endswith(('.png', '.jpg', '.jpeg'))]

# 만약 이미지가 1000개 이하이면 모든 이미지 선택
num_images_to_select = min(1000, len(image_files))

# 이미지 파일을 무작위로 선택
selected_images = random.sample(image_files, num_images_to_select)

# 선택된 이미지를 복사하여 저장
for image_path in selected_images:
    image_name = os.path.basename(image_path)
    output_path = os.path.join(dest_filepath, image_name)
    shutil.copy(image_path, output_path)

print(f'{len(selected_images)} images selected and saved to {dest_filepath}.')
