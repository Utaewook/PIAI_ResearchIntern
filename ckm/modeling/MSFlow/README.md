# MSFlow 모델 구현


---
### 모델 구조
![framework.png](imgs%2Fframework.png)
- 공식 github: [cool-xuan/msflow](https://github.com/cool-xuan/msflow)

---

### requirements
- 공식 github에서의 requirements
  - Python 3.9
  - scikit-learn
  - scikit-image
  - Pytorch >= 1.10
  - CUDA 11.3
  - [FrEIA](https://github.com/VLL-HD/FrEIA) (Please install FrEIA following the [official installation](https://github.com/VLL-HD/FrEIA#table-of-contents))   
  
   
- 실제 구축 환경: [env_setting.txt](env_setting.txt)
  - Python 3.9.18
  - scikit-learn 1.3.0
  - scikit-image 0.19.3
  - Pytorch 2.1
    - Pytorch와 관련된 패키지의 경우에는 [공식 홈페이지](https://pytorch.org/get-started/locally/)에서 데스크탑 환경에 맞는 os와 cuda 버전을 선택하여 설치
  - CUDA 12.1
  - FrEIA
  - GPUtil 1.4.0 
  - matplotlib 
  - opencv
  
  
 
### 디렉토리

- data: 데이터셋이 포함된 디렉토리
- gpu_monitoring: test 시 gpu 사용량 모니터링 결과를 저장하는 디렉토리
- models: extractor, parallel flows, fusion flow 모델이 구현된 코드 디렉토리
- segmentation: train.py 파일에서 생성한 segmentation 이미지, anomaly map, 원본 이미지가 저장된 디렉토리 
- work_dirs: train 완료된 모델 구조가 저장된 디렉토리 (.pt 파일)


### 코드별 설명

- _**datasets.py: 환편기 데이터셋 클래스가 선언된 코드 -> 카메라 스트림을 가져와서 데이터셋으로 생성하는 코드 구현 필요**_
- _**default.py: train, test에 필요한 파라미터가 선언 되어 있는 코드. 다른 코드에서 import default as c로 선언되어 사용됨**_ 
- evaluations.py: AUROC와 같은 평가 함수들이 선언 되어 있는 코드.
- finetuning.py: 파인튜닝을 위한 코드가 선언되어 있는 파일
- finetuning_dataset_make.py: 파인튜닝을 위한 데이터셋을 생성하는 코드가 선언되어 있는 파일 (기존의 train 데이터 셋에서 무작위로 1000장 선택) 
- gpu_ex.ipynb: GPU 성능 확인을 위해 test command를 반복적으로 실행하는 자동화 코드. command를 설정하고, main.py에서 실행된 gpu 모니터링 결과를 저장함
- inference.py: 실제 환편기에 적용하기 위해 모델 pt 파일을 가져와서 추론만을 수행하는 코드 (작성 필요)
- _**main.py: train, test를 실행하는 메인 코드. args를 입력받아 실행함. 멀티스레드로 GPU 모니터링을 하는 기능 또한 포함**_
- _**post_process.py: 모델에서 나온 출력 값을 anomaly map, score로 후처리 하는 코드**_
- pruning.py: 가지치기를 위한 코드 (작성 필요)
- temp.ipynb: test 과정에서 추론한 결과와, label값을 비교하고 분석한 코드
- _**train.py: train, test를 위한 전반적인 기능이 구현된 코드**_
- _**utils.py: train, test에 필요한 함수들이 선언되어 있는 코드. 모델의 가중치를 가져오고, 저장하는 기능과 segmentation하는 기능이 포함**_
- weight_analysis.ipynb: 모델의 가중치를 분석하는 코드


### train, test 실행 방법
1. anaconda 가상 환경을 생성 후, MSFlow 프로젝트가 있는 위치로 이동.
2. train, test에 사용될 데이터 셋을 구축 후 해당 경로를 default.py의 c.data_path에 입력(혹은 코드 레벨에서 직접 수정)
3. train, test 상황에 따라 args를 바꿔가며 실행. 예시는 아래와 같음.
- train
  ```shell
  python main.py --mode train --extractor resnet18 --batch-size 8 --gpu 0
  ```
- test
  ```shell
  python main.py --mode test --eval_ckpt E:\codes\ckm\modeling\MSFlow\work_dirs\msflow_resnet34_avgpool_pl258\textile\best_det.pt --extractor resnet34 --batch-size 400 --gpu 1
  ```

### finetuning 과정을 통한 새로운 데이터셋 학습 방법
1. finetuning_dataset_make.py를 실행하여 파인튜닝을 위한 데이터셋을 생성 or 새로운 데이터셋을 생성
2. 해당 데이터 셋의 디렉토리 경로를 finetuning.py의 c.data_path에 입력
3. 상황에 따라, pretrain 모델을 사용하려면 finetuning.py의 ckpt_path를 지정
4. c.[arg_name]을 지정해 train 파라미터(예를 들면 epoch, batch size 등)을 설정
5. finetuning.py를 실행하여 파인튜닝을 수행


---