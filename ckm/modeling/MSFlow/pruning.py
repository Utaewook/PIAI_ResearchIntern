import torch

import default as c
from utils import load_weights, save_weights
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from finetuning import finetuning


def prune_parallel_flows(parallel_flows, pruning_rate):
    '''
    MSFlow 모델의 병렬 흐름 파라미터를 프루닝합니다.

    Args:
    parallel_flows (list of nn.Module): 모델의 parallel_flows.
    pruning_rate (float): 프루닝할 가중치의 비율 (0에서 1 사이).
    '''
    for parallel_flow in parallel_flows:
        for name, param in parallel_flow.named_parameters():
            if 'weight' in name:  # 가중치만 프루닝한다고 가정
                # 비구조적 프루닝 적용
                threshold = torch.quantile(torch.abs(param.data), pruning_rate)
                mask = torch.abs(param.data) > threshold
                param.data *= mask.float()
            print(name, param, 'pruning done')
    print('pruning done')

if __name__ == '__main__':
    pt_path = 'E:\\codes\\ckm\\modeling\\MSFlow\\work_dirs\\msflow_resnet34_avgpool_pl258\\textile\\best_det.pt'  # resnet 34 모델 경로
    c.extractor = 'resnet34'

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
    load_weights(parallel_flows, fusion_flow, pt_path)

    pruning_rate = 0.5  # 프루닝 비율

    prune_parallel_flows(parallel_flows, pruning_rate)

    save_weights(0,parallel_flows, fusion_flow, f'pruned34_{pruning_rate}','E:\\codes\\ckm\\modeling\\MSFlow\\work_dirs\\pruned')