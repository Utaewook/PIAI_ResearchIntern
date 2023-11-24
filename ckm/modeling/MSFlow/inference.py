# import cv2
# import torch.utils.data
# from torchvision import transforms as T
# from torchvision.transforms import InterpolationMode

# from post_process import post_process
# from datasets import ImageStreamDataset, CKMDataset
# from models.extractors import build_extractor
# from models.flow_models import build_msflow_model

# from train import model_forward

# from PIL import Image

# import numpy as np


# def model_usage(c):
#     predict_module = MSFlowPredictModule(c)

#     # 예시 이미지
#     # img_path = 'C:\\Users\\piai\\Desktop\\codes\\modeling\\MSFlow\\data\\CKM\\test\\bad\\0_782.png'
#     #
#     # img = Image.open(img_path).convert("RGB")
#     #
#     # lbl, score = predict_module.predict_one(img)
#     #
#     # img_name = img_path.split('\\')[-1]
#     # print(f'{img_name} is {lbl} and anomaly score is {score}')

#     # # 만약 카메라를 활용해서 데이터셋을 생성한 후 예측을 진행하는 경우
#     # cap = '카메라 객체'
#     # ds = ImageStreamDataset(cap)

#     # # 배치사이즈는 변경해야함
#     # loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
#     # predict_module.predict(loader)

#     # 테스트 데이터 셋으로 진행
#     test_dataset = CKMDataset(c, is_train=False)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False,
#                                               num_workers=c.workers, pin_memory=True)

#     predict_module.predict(test_loader)


# class MSFlowPredictModule():
#     def __init__(self, c):
#         # args 저장
#         self.c = c

#         # build model
#         self.extractor, output_channels = build_extractor(self.c)
#         self.extractor = self.extractor.to(self.c.device).eval()

#         self.parallel_flows, self.fusion_flow = build_msflow_model(self.c, output_channels)
#         self.parallel_flows = [parallel_flow.to(self.c.device) for parallel_flow in self.parallel_flows]
#         self.fusion_flow = self.fusion_flow.to(self.c.device)

#         # weight loading
#         self.load_model_weight(self.c.eval_ckpt)

#         # img processing
#         self.transform_img = T.Compose([
#             T.Resize(c.input_size, InterpolationMode.LANCZOS),
#             T.ToTensor()])

#         self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

#     def load_model_weight(self, path):
#         '''
#             입력받은 경로에 존재하는 .pt, .pth 파일의 가중치를 가져와
#             build한 모델에 적용
#         '''
#         print('Load weights from {}'.format(path))

#         state_dict = torch.load(path)

#         fusion_state = state_dict['fusion_flow']
#         maps = {}
#         for i in range(len(self.parallel_flows)):
#             maps[self.fusion_flow.module_list[i].perm.shape[0]] = i

#         temp = {}

#         for k, v in fusion_state.items():
#             if 'perm' not in k:
#                 continue
#             temp[k.replace(k.split('.')[1], str(maps[v.shape[0]]))] = v

#         for k, v in temp.items():
#             fusion_state[k] = v

#         self.fusion_flow.load_state_dict(fusion_state, strict=False)

#         for parallel_flow, state in zip(self.parallel_flows, state_dict['parallel_flows']):
#             parallel_flow.load_state_dict(state, strict=False)

#     def predict(self, loader):
#         '''
#             이미지 스트림(queue 형태의 batch)을 입력 받고,
#             입력 받은 이미지의 모델 예측 라벨을 반환해주는
#             함수
#         '''
#         outputs_list = [list() for _ in self.parallel_flows]
#         size_list = []

#         seg_path = '.\\anomaly_maps'

#         with torch.no_grad():
#             for idx, (img, _) in enumerate(loader):
#                 img = img.to(self.c.device)
#                 z_list, jac = model_forward(self.c, self.extractor, self.parallel_flows, self.fusion_flow, img)
#                 loss = 0.
#                 for lvl, z in enumerate(z_list):
#                     if idx == 0:
#                         size_list.append(list(z.shape[-2:]))
#                     logp = - 0.5 * torch.mean(z ** 2, 1)
#                     outputs_list[lvl].append(logp)
#                     loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))

#             # 아래의 map, score를 활용해서 anomaly contour 이미지 생성해야함
#             anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(self.c, size_list, outputs_list)

#             for idx, (img, label) in enumerate(loader):

#                 contour = self.find_contours(anomaly_score_map_add[idx])
#                 overlay

#                 predict_anomaly = 'bad' if anomaly_score[idx] > self.c.threshold else 'good'

#                 if 25 < idx < len(loader) - 25:
#                     # 일단 50씩만 저장해서 보기
#                     continue

#                 save_img = Image.fromarray(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
#                 save_img.save(seg_path + f"\\{predict_anomaly}_{label[0]}_{idx}.png")


#     # 이상치 탐지 맵에서 contour를 찾아주는 함수
#     def find_contours(self, anomaly_map, threshold=0.5):
#         # 이상치 맵에서 이상치 영역을 찾습니다.
#         _, binary_map = cv2.threshold(anomaly_map, threshold, 1, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(np.uint8(binary_map), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         return contours

#     def overlay_contours_on_image(self, original_image, contours, output_path):
#         # 이미지 위에 contour를 그립니다.
#         output_image = original_image.copy()
#         cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)  # 빨간색으로 contour를 그립니다.

#         # 결과 이미지를 저장합니다.
#         cv2.imwrite(output_path, output_image)









#     # def segment_anomalies(self, anomaly_map, original_image, threshold=0.5):
#     #     binary_map = (anomaly_map > threshold).astype(np.uint8)
#     #
#     #     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
#     #
#     #     # 원본 이미지에 이상치를 표시 (예: 빨간색 상자)
#     #
#     #     result_image = original_image  # 이미지 객체 그대로 사용
#     #
#     #     # 'cv2.RETR_EXTERNAL' 플래그를 사용하여 외부 윤곽선만 찾습니다.
#     #     contours, _ = cv2.findContours(binary_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     #     cv2.drawContours(np.array(result_image), contours, -1, (0, 0, 255), 2)  # 빨간색 외곽선
#     #
#     #     return result_image
#     #
#     # def t2img(self, t):
#     #     t = (t * 255).byte()
#     #     if t.dim() == 4:
#     #         t = t[0]  # 배치 차원 제거
#     #     return Image.fromarray(t.permute(1, 2, 0).numpy())  # 텐서를 이미지로 변환

#     # def predict_one(self, img):
#     #     '''
#     #         하나의 이미지만 모델에 넣어 예측 결과를 반환해주는 함수
#     #     '''
#     #     outputs_list = [list() for _ in self.parallel_flows]
#     #     size_list = []
#     #
#     #     img = self.normalize(self.transform_img(img))
#     #     img_shape = list(img.shape)
#     #     print(img_shape)
#     #     img = img.reshape(1, *img_shape)
#     #     with torch.no_grad():
#     #         flag = True
#     #
#     #         img.to(self.c.device)
#     #
#     #         z_list, jac = model_forward(self.c, self.extractor, self.parallel_flows, self.fusion_flow, img)
#     #         loss = 0.
#     #         for lvl, z in enumerate(z_list):
#     #             if flag:
#     #                 size_list.append(list(z.shape[-2:]))
#     #                 flag = False
#     #             logp = - 0.5 * torch.mean(z ** 2, 1)
#     #             outputs_list[lvl].append(logp)
#     #             loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))
#     #     anomaly_score, _, _ = post_process(self.c, size_list, outputs_list)
#     #
#     #     return 'bad' if anomaly_score > self.c.threshold else 'good', anomaly_score


# """
# 이미지 데이터를 iterate하게 혹은 하나씩 읽어와서 클래스 예측값을 리턴해주는 클래스를 생성해볼것

# inference_meta_epoch, post_process 함수 보고, 클래스 내부 메소드로 생성 해볼것
# """
