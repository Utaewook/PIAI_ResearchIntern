import torch
import os
import threading
from needmodule import ImageFolderWithPath, visualize_one_sample, gpu_memory,gpu_percentage
from torchvision import transforms

path = 'E:\\codes\\ckm\\modeling\\EfficientAD_S(module)_v1025\\output\\small_v1_10_25\\trainings\\mvtec_ad' #저장된 모델 불러오는 코드
auto = torch.load(path+'/autoencoder_final.pth')
stu = torch.load(path+'/student_final.pth')
tea = torch.load(path+'/teacher_final.pth')
data_path = 'E:\\codes\\ckm\\modeling\\EfficientAD_S(module)_v1025\\Data_test' # test하고 싶은 폴더명 지정하시면 됩니다. 하지만 폴더안에 폴더를 하나 더 만들고 이미지 파일 삽입하셔야합니다.
image_size = 256
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
if __name__ == '__main__':

    exit_flag = threading.Event()

    gpu_percentage_thread = threading.Thread(
        target=gpu_percentage,
        args=(exit_flag,)
    )
    gpu_percentage_thread.start()

    gpu_usage_thread = threading.Thread(
        target=gpu_memory,
        args=(exit_flag,)
    )
    gpu_usage_thread.start()



    test_set = ImageFolderWithPath(
        os.path.join(data_path))
    list_a = []
    for image, _, _ in test_set:
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        image = image.cuda()
        a = visualize_one_sample(image, orig_height, orig_width, auto, stu, tea)
        print(a)



    exit_flag.set()
    gpu_percentage_thread.join()
    gpu_usage_thread.join()
