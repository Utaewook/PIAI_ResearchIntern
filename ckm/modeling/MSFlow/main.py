import os, random
import numpy as np
import torch
import argparse
import GPUtil
import time
import threading

from train import train

def init_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parsing_args(c):
    parser = argparse.ArgumentParser(description='msflow')
    parser.add_argument('--mode', default='train', type=str, 
                        help='train, test or use.')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='resume training or not.')
    parser.add_argument('--eval_ckpt', default='', type=str, 
                        help='checkpoint path for evaluation.')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--class-name', default='textile', type=str)
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--batch-size', default=8, type=int, 
                        help='train batch size')
    parser.add_argument('--meta-epochs', default=25, type=int,
                        help='number of meta epochs to train')
    parser.add_argument('--sub-epochs', default=4, type=int,
                        help='number of sub epochs to train')
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str, 
                        help='feature extractor')
    parser.add_argument('--pool-type', default='avg', type=str, 
                        help='pool type for extracted feature maps')
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, metavar='L', nargs='+',
                        help='number of flow blocks used in parallel flows.')
    parser.add_argument('--pro-eval', action='store_true', default=False, 
                        help='evaluate the pro score or not.')
    parser.add_argument('--pro-eval-interval', default=4, type=int, 
                        help='interval for pro evaluation.')
    parser.add_argument('--threshold', default=0.925, type=float,
                        help='threshold of score classify.')

    args = parser.parse_args()

    for k, v in vars(args).items():
        setattr(c, k, v)
        
    if c.class_name == 'transistor':
        c.input_size = (256, 256)
    elif c.class_name == 'textile':
        c.input_size = (128, 128)
    else:
        c.input_size = (512, 512)

    return c

def inspect_args(mode):
    import importlib
    import inspect

    # 모듈 이름
    module_name = "default"  # 실제 모듈 이름으로 변경해야 합니다

    # 모듈을 동적으로 import
    module = importlib.import_module(module_name)

    # 모듈 내의 변수 및 함수 목록 저장
    with open('args_{}.txt'.format(mode),'w') as f:
        for name, obj in inspect.getmembers(module):
            if not name.startswith("__"):  # 이중 언더스코어로 시작하는 변수는 제외
                f.write(f"{name} = {obj}\n")

def gpu_percentage(exit_flag):
    gpu_usage_list = []

    while not exit_flag.is_set():
        try:
            gpu_usage = GPUtil.getGPUs()[0].load * 100.0
            gpu_usage_list.append(gpu_usage)

            time.sleep(0.5)
        except Exception as e:
            print(f"Error while monitoring GPU usage: {e}")

    if gpu_usage_list:
        min_usage = min(gpu_usage_list)
        max_usage = max(gpu_usage_list)
        avg_usage = sum(gpu_usage_list) / len(gpu_usage_list)

        print("\nGPU percentage usage")
        print(f"Minimum GPU usage: {min_usage}%")
        print(f"Maximum GPU usage: {max_usage}%")
        print(f"Average GPU usage: {avg_usage}%\n")

def gpu_memory(exit_flag):
    gpu_memory_list = []

    while not exit_flag.is_set():
        try:
            gpu_memory_kb = GPUtil.getGPUs()[0].memoryUsed
            gpu_memory_mb = gpu_memory_kb / 1024  # KB를 MB로 변환
            gpu_memory_list.append(gpu_memory_mb)

            time.sleep(0.5)
        except Exception as e:
            print(f"Error while monitoring GPU memory usage: {e}")

    if gpu_memory_list:
        min_memory = min(gpu_memory_list)
        max_memory = max(gpu_memory_list)
        avg_memory = sum(gpu_memory_list) / len(gpu_memory_list)

        print("\nGPU memory usage")
        print(f"Minimum GPU memory usage: {min_memory:.2f} MB")
        print(f"Maximum GPU memory usage: {max_memory:.2f} MB")
        print(f"Average GPU memory usage: {avg_memory:.2f} MB\n")

def main(c):
    c = parsing_args(c)
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    init_seeds(seed=c.seed)
    c.version_name = 'msflow_{}_{}pool_pl{}'.format(c.extractor, c.pool_type, "".join([str(x) for x in c.parallel_blocks]))
    c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.class_name)

    # inspect_args(c.mode)

    if c.mode == 'use':
        from inference import model_usage
        model_usage(c)
        return

    train(c)

if __name__ == '__main__':
    import default as c
    # exit_flag = threading.Event()

    # gpu_percentage_thread = threading.Thread(
    #     target=gpu_percentage,
    #     args=(exit_flag,)
    # )
    # gpu_percentage_thread.start()

    # gpu_usage_thread = threading.Thread(
    #     target=gpu_memory,
    #     args=(exit_flag,)
    # )
    # gpu_usage_thread.start()

    main(c)

    # exit_flag.set()
    # gpu_percentage_thread.join()
    # time.sleep(1)
    # gpu_usage_thread.join()