import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tifffile
import cv2

from datasets import CKMDataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights, segmentation
from evaluations import eval_det_loc, eval_det_loc_only


def model_forward(c, extractor, parallel_flows, fusion_flow, image):
    # print(f'mode-{c.mode}  image info-{image.shape}')
    h_list = extractor(image)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_list = []
    parallel_jac_list = []
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, parallel_flows, c.c_conds)):
        y = pool_layer(h)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond, ])
        z_list.append(z)
        parallel_jac_list.append(jac)

    z_list, fuse_jac = fusion_flow(z_list)
    jac = fuse_jac + sum(parallel_jac_list)

    return z_list, jac


def train_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler):
    parallel_flows = [parallel_flow.train() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.train()
    iters = len(loader)
    for sub_epoch in range(c.sub_epochs):
        epoch_loss = 0.
        image_count = 0
        for idx, (image, _) in enumerate(loader):
            image = image.to(c.device)
            z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)

            loss = 0.
            for z in z_list:
                loss += 0.5 * torch.sum(z**2, (1, 2, 3))
            loss = loss - jac
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2)
            optimizer.step()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if warmup_scheduler:
            warmup_scheduler.step()
        if decay_scheduler:
            decay_scheduler.step()

        mean_epoch_loss = epoch_loss / image_count
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                epoch, sub_epoch, mean_epoch_loss, lr))


# 주어진 이미지 데이터로 부터 이상치 감지 모델을 사용해 이상치를 예측하고 결과 반환 하는 함수
def inference_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow):
    # 모델들을 evaluation 모드로 변경
    parallel_flows = [parallel_flow.eval() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.eval()

    # epoch 및 image_count 초기화
    epoch_loss = 0.
    image_count = 0
    gt_label_list = list()
    # gt_mask_list = list()
    outputs_list = [list() for _ in parallel_flows]
    size_list = []
    start = time.time()
    with torch.no_grad():
        # 로더를 순회하며 이미지와 라벨을 가져옴
        for idx, (image, label) in enumerate(loader):
            image = image.to(c.device)

            # 라벨값을 np로 저장
            gt_label_list.extend(t2np(label))
            # gt_mask_list.extend(t2np(mask))

            # 이미지를 모델이 전달하고, 각 병렬 플로우와 퓨전 플로우에서 추출된 출력인 z_list와 야코비안 값을 얻는다.
            z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)

            # loss 초기화 하고 z_list의 각 레벨에서 로그 확률값을 계산하고 ouputlist에 추가한다. 또한 KL Divergence를 포함한 손실 계산
            loss = 0.
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                logp = - 0.5 * torch.mean(z**2, 1)
                outputs_list[lvl].append(logp)
                loss += 0.5 * torch.sum(z**2, (1, 2, 3))

            loss = loss - jac
            loss = loss.mean()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]

        # 모든 레벨에 대한 손실을 합산하고, 야코비안을 뺀 후 평균을 구한다 = 이미지에 대한 손실
        mean_epoch_loss = epoch_loss / image_count
        fps = len(loader.dataset) / (time.time() - start)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                epoch, mean_epoch_loss, fps))


    """
    gt_label_list = 각 이미지의 실제 레이블을 저장하는 리스트
    outputs_list = 모델의 각 parallel flow에서 생성된 이상치 확률값을 저장하는 리스트
    size_list = 각 이미지에서 계산된 이상치 확률 맵의 크기를 저장하는 리스트
    """
    return gt_label_list, outputs_list, size_list


def train(c):
    print(c)

    # 모델 build
    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)  # MSFlow 모델 생성
    parallel_flows = [parallel_flow.to(c.device) for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.to(c.device)
    params = list(fusion_flow.parameters())
    for parallel_flow in parallel_flows:
        params += list(parallel_flow.parameters())

    optimizer = torch.optim.Adam(params, lr=c.lr)

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)
    # loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    # loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)

    start_epoch = 0

    # test 데이터 셋 load
    test_dataset = CKMDataset(c, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False,
                                              num_workers=c.workers, pin_memory=True)


    # test모드로 시작될 시 test만 수행 후 함수 종료
    if c.mode == 'test':

        start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)
        epoch = start_epoch + 1
        gt_label_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
        best_det_auroc = eval_det_loc_only(det_auroc_obs, epoch, gt_label_list, anomaly_score)


        for i, (src, anomaly_map) in enumerate(zip(test_dataset, anomaly_score_map_add)):
            img,_ = src
            
            seg_img = segmentation(img,anomaly_map)

            normalized_image = (img - img.min()) / (img.max() - img.min())
            original_image_np = (normalized_image.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            original_image_np = np.ascontiguousarray(original_image_np)


            plt.imsave(f'./segmentation/src_image/src_{i}.png', original_image_np)
            plt.imsave(f'./segmentation/map/map_{i}.png', anomaly_map)
            plt.imsave(f"./segmentation/seg_image/{'bad'if anomaly_score[i] > c.threshold else 'good'}_{i}_seg.jpeg",seg_img)
            
            


        # anomaly score 및 라벨값 저장
        # with open('det_score.txt', 'w') as f:
        #     for s, gt in zip(anomaly_score, gt_label_list):
        #         f.write(f'{gt}\t{s}\n')
        #
        # os.chdir('anomaly_maps')
        #
        # import matplotlib.pyplot as plt
        # for i, (a, m, gt) in enumerate(zip(anomaly_score_map_add, anomaly_score_map_mul,gt_label_list)):
        #     plt.imsave(f"{'bad'if gt==1 else 'good'}_{i}_map_add.jpeg", a)
        #     plt.imsave(f"{'bad'if gt==1 else 'good'}_{i}_map_mul.jpeg", m)

        print(f"test det auroc: {best_det_auroc}\nanomaly score: {anomaly_score}, len({len(anomaly_score)})")


        # print(f'lengths:\n\tanomaly score: {len(anomaly_score)}\n\tanomaly map add: {len(anomaly_score_map_add)}\n\tanomaly map mul: {len(anomaly_score_map_mul)}')
        # best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)

        return

    # train 데이터 셋 load
    train_dataset = CKMDataset(c, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True,num_workers=c.workers, pin_memory=True)


    if c.resume:
        last_epoch = load_weights(parallel_flows, fusion_flow, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
        start_epoch = last_epoch + 1
        print('Resume from epoch {}'.format(start_epoch))

    if c.lr_warmup and start_epoch < c.lr_warmup_epochs:
        if start_epoch == 0:
            start_factor = c.lr_warmup_from
            end_factor = 1.0
        else:
            start_factor = 1.0
            end_factor = c.lr / optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=(c.lr_warmup_epochs - start_epoch)*c.sub_epochs)
    else:
        warmup_scheduler = None

    mile_stones = [milestone - start_epoch for milestone in c.lr_decay_milestones if milestone > start_epoch]

    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma)
    else:
        decay_scheduler = None


    # train epoch 시작
    for epoch in range(start_epoch, c.meta_epochs):
        print()
        train_meta_epoch(c, epoch, train_loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler)

        gt_label_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

        # if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
        #     pro_eval = True
        # else:
        #     pro_eval = False

        # best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, pro_eval)

        best_det_auroc = eval_det_loc_only(det_auroc_obs, epoch, gt_label_list, anomaly_score)


        save_weights(epoch, parallel_flows, fusion_flow, 'last', c.ckpt_dir, optimizer)
        if best_det_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_det', c.ckpt_dir)
        # if best_loc_auroc and c.mode == 'train':
        #     save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
        # if best_loc_pro and c.mode == 'train':
        #     save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_pro', c.ckpt_dir)
