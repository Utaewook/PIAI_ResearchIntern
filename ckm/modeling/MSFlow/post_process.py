import numpy as np
import torch
import torch.nn.functional as F


# 주어진 입력에 대한 이상치 감지 결과를 계산하고 반환하는 함수
def post_process(c, size_list, outputs_list):
    print('Multi-scale sizes:', size_list)

    # logp_maps: 다중 스케일의 로그 확률 맵들이 저장
    # prop_maps: 속성 맵들이 저장
    logp_maps = [list() for _ in size_list]
    prop_maps = [list() for _ in size_list]

    for l, outputs in enumerate(outputs_list):
        # output = torch.tensor(output, dtype=torch.double)
        outputs = torch.cat(outputs, 0)
        logp_maps[l] = F.interpolate(outputs.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
        output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        del outputs
        prob_map = torch.exp(output_norm) # convert to probs in range [0:1]
        del output_norm
        prop_maps[l] = F.interpolate(prob_map.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)

    # logp_map: 모든 스케일에서 계산한 로그 확률 맵을 합산한 값
    logp_map = sum(logp_maps)
    del logp_maps

    # logp_map 값을 정규화 해준다
    logp_map -= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]

    # prop_map_mul 값을 지정(logp_map을 속성 맵으로 변환 --> 0 과 1사이의 확률값으로 바뀜)
    prop_map_mul = torch.exp(logp_map)
    del logp_map

    # prop_map_mul의 최대값을 찾아서, 해당 위치에서 최대값을 뺀 결과를 저장
    # anomaly_score_map_mul은 이미지의 이상치 점수를 나타내는 맵
    anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
    del prop_map_mul
    batch = anomaly_score_map_mul.shape[0]
    top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)

    # anomaly_score 산출 anomaly_score_map_mul을 평균 내서!
    anomaly_score = np.mean(
        anomaly_score_map_mul.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)


    # prop_map값을 단순 합산 해서 생성
    prop_map_add = sum(prop_maps)
    del prop_maps
    prop_map_add = prop_map_add.detach().cpu().numpy()

    # prop_map_add의 최대값과 현재 위치를 뺀 값을 저장해 이미지의 해당 위치에서의 anomaly_score 산출
    anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add
    del prop_map_add

    # 이상 점수, 이상 점수 맵(덧셈 매트릭스), 이상 점수 맵(곱셈 매트릭스)
    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()