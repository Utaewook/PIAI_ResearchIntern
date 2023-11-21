import torch
import numpy as np
import os

import default as c
from utils import Score_Observer, load_weights, save_weights
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from datasets import CKMDataset
from post_process import post_process
from evaluations import eval_det_loc_only
from train import train_meta_epoch, inference_meta_epoch

def finetuning():
    pass

c.mode = 'train'
c.data_path = '../../dataset/CKM_finetuning'
ckpt_path = 'E:\\codes\\ckm\\modeling\\MSFlow\\work_dirs\\msflow_wide_resnet50_2_avgpool_pl258\\textile\\best_det.pt'

# model build
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

train_dataset = CKMDataset(c, is_train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

test_dataset = CKMDataset(c, is_train=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)


start_epoch = 0

_ = load_weights(parallel_flows, fusion_flow, ckpt_path, optimizer)


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


    save_weights(epoch, parallel_flows, fusion_flow, 'finetuned_last', c.ckpt_dir, optimizer)
    if best_det_auroc and c.mode == 'train':
        save_weights(epoch, parallel_flows, fusion_flow, 'finetuned_best_det', c.ckpt_dir)