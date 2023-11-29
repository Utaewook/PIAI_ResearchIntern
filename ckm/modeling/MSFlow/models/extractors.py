from .resnet import *


# Resnet 모델 중 하나를 생성하고, 해당 모델의 각 레이어에서 출력 채널 수를 추출하는 함수
def build_extractor(c):

    # c에 있는 매개변수에 따라 resnet모델 생성
    if   c.extractor == 'resnet18':
        extractor = resnet18(pretrained=True, progress=True)
    elif c.extractor == 'resnet34':
        extractor = resnet34(pretrained=True, progress=True)
    elif c.extractor == 'resnet50':
        extractor = resnet50(pretrained=True, progress=True)
    elif c.extractor == 'resnext50_32x4d':
        extractor = resnext50_32x4d(pretrained=True, progress=True)
    elif c.extractor == 'wide_resnet50_2':
        extractor = wide_resnet50_2(pretrained=True, progress=True)


    # 생성한 모델의 output channel들을 저장할 리스트
    output_channels = []

    # resnet의 채널 수를 리스트에 저장
    if 'wide' in c.extractor:
        for i in range(3):
            output_channels.append(eval('extractor.layer{}[-1].conv3.out_channels'.format(i+1)))
    else:
        for i in range(3):
            # output_channels.append(getattr(extractor, 'layer{}'.format(i+1))[-1].conv2.out_channels)
            # output_channels.append(extractor.eval('layer{}'.format(i+1))[-1].conv2.out_channels)
            output_channels.append(getattr(extractor, f'layer{i+1}')[-1].conv2.out_channels)


    # 생성한 extractor 객체와 채널 수를 저장한 리스트를 반환
    print("Channels of extracted features:", output_channels)
    return extractor, output_channels