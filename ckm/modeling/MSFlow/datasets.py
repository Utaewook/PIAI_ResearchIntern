import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

__all__ = ('MVTecDataset', 'CKMDataset',)

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                     'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                     'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

BOOLEAN_CLASS_NAMES = ['textile']


class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name,
                                                                                           MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # x = Image.open(x).convert('RGB')
        x = Image.open(x)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)



# 환편기 데이터들을 데이터 셋 객체로 변환
class CKMDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in BOOLEAN_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name,
                                                                                             BOOLEAN_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size

        # load dataset
        self.x, self.y = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([
            T.Resize(c.input_size, InterpolationMode.LANCZOS),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

        # print
        print(f"loaded image dataset: {len(self)}, type: {type(self.x[0])}")

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = Image.open(x).convert('RGB')
        x = self.normalize(self.transform_x(x))
        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []

        img_dir = os.path.join(self.dataset_path, phase)

        img_types = sorted(os.listdir(img_dir))

        # print(img_types)
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f != 'desktop.ini'])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)

    def img_process(self, img):
        return self.normalize(self.transform_x(img))


# 카메라 혹은 이미지 스트림을 가져와서 데이터 셋으로 만듬
class ImageStreamDataset(Dataset):
    def __init__(self, c, stream):
        self.input_size = c.input_size

        self.stream = stream

        # set transforms
        self.transform_frame = T.Compose([
            T.Resize(c.input_size, InterpolationMode.LANCZOS),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        ret, frame = self.stream.read()

        if not ret:
            raise StopIteration

        frame = self.normalize(self.transform_frame(frame))

        return frame

    def __len__(self):
        return float('inf')