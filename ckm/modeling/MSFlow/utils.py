import os
import math
import datetime
import cv2

import numpy as np
import torch

# 2차원 위치 인코딩 생성
# D: 모델의 차원
# H: 위치의 높이, W: 위치의 너비
# return: DxHxW 위치 행렬
def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


# train 과정에서 모델의 가중치를 저장하는 함수
# epoch: train 할 때의 현재 epoch
def save_weights(epoch, parallel_flows, fusion_flow, model_name, ckpt_dir, optimizer=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    file_name = '{}.pt'.format(model_name)
    file_path = os.path.join(ckpt_dir, file_name)
    print('Saving weights to {}'.format(file_path))
    state = {'epoch': epoch,
             'fusion_flow': fusion_flow.state_dict(),
             'parallel_flows': [parallel_flow.state_dict() for parallel_flow in parallel_flows]}
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    torch.save(state, file_path)


# 주어진 경로에서 모델의 가중치 파일을 불러옴(.pt)
def load_weights(parallel_flows, fusion_flow, ckpt_path, optimizer=None):
    print('Loading weights from {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    
    fusion_state = state_dict['fusion_flow']
    maps = {}
    for i in range(len(parallel_flows)):
        maps[fusion_flow.module_list[i].perm.shape[0]] = i
    temp = dict()
    for k, v in fusion_state.items():
        if 'perm' not in k:
            continue
        temp[k.replace(k.split('.')[1], str(maps[v.shape[0]]))] = v
    for k, v in temp.items():
        fusion_state[k] = v
    fusion_flow.load_state_dict(fusion_state, strict=False)

    for parallel_flow, state in zip(parallel_flows, state_dict['parallel_flows']):
        parallel_flow.load_state_dict(state, strict=False)

    if optimizer:
        optimizer.load_state_dict(state_dict['optimizer'])

    return state_dict['epoch']


# 입력받은 이미지와 anomaly map을 이용하여 contour를 찾고, 원본 이미지에 contour를 그린 이미지를 반환하는 함수
def segmentation(src_image, anomaly_map, threshold=0.7):
    # 텐서를 넘파이 배열로 변환
    # src_image = (src_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    normalized_image = (src_image - src_image.min()) / (src_image.max() - src_image.min())
    original_image_np = (normalized_image.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    original_image_np = np.ascontiguousarray(original_image_np)

    # 이미지를 0-255 범위로 normalize
    anomaly_map_normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 이미지 이진화 (binary thresholding)
    ret, thresh = cv2.threshold(anomaly_map_normalized, 127, 255, cv2.THRESH_BINARY)


    # 이진화된 맵을 사용하여 contour를 찾음
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 원본 이미지에 contour를 그림
    segmented_image = original_image_np.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)  # 초록색으로 contour를 그림
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2RGB)
    return segmented_image


class Score_Observer:
    def __init__(self, name, total_epochs):
        self.name = name
        self.total_epochs = total_epochs
        self.max_epoch = 0
        self.max_score = 0.0
        self.last_score = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        best = False
        if score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            best = True
        if print_score:
            self.print_score(epoch)
        
        return best

    def print_score(self, epoch):
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"), 
            'Epoch [{:d}/{:d}] {:s}: last: {:.2f}\tmax: {:.2f}\tepoch_max: {:d}'.format(
                epoch, self.total_epochs-1, self.name, self.last, self.max_score, self.max_epoch))

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None