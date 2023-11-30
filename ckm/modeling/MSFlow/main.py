import os, random
import numpy as np
import torch
import argparse
import GPUtil
import time
import threading
import queue

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

    # 파라미터 모듈 이름
    module_name = "default"

    # 모듈을 동적으로 import
    module = importlib.import_module(module_name)

    # 모듈 내의 변수 및 함수 목록 저장
    with open('args_{}.txt'.format(mode),'w') as f:
        for name, obj in inspect.getmembers(module):
            if not name.startswith("__"):  # 이중 언더스코어로 시작하는 변수는 제외
                f.write(f"{name} = {obj}\n")

# def gpu_percentage(exit_flag, gpu):
#     gpu_usage_list = []

#     while not exit_flag.is_set():
#         try:
#             gpu_usage = GPUtil.getGPUs()[gpu].load * 100.0
#             gpu_usage_list.append(gpu_usage)

#             time.sleep(0.5)
#         except Exception as e:
#             print(f"Error while monitoring GPU usage: {e}")

#     if gpu_usage_list:
#         min_usage = min(gpu_usage_list)
#         max_usage = max(gpu_usage_list)
#         avg_usage = sum(gpu_usage_list) / len(gpu_usage_list)

#         print("\nGPU percentage usage")
#         print(f"Minimum GPU usage: {min_usage}%")
#         print(f"Maximum GPU usage: {max_usage}%")
#         print(f"Average GPU usage: {avg_usage}%\n")

# def gpu_memory(exit_flag, gpu):
#     gpu_memory_list = []

#     while not exit_flag.is_set():
#         try:
#             gpu_memory_kb = GPUtil.getGPUs()[gpu].memoryUsed
#             gpu_memory_mb = gpu_memory_kb / 1024  # KB를 MB로 변환
#             gpu_memory_list.append(gpu_memory_mb)

#             time.sleep(0.5)
#         except Exception as e:
#             print(f"Error while monitoring GPU memory usage: {e}")

#     time.sleep(0.4)
#     if gpu_memory_list:
#         min_memory = min(gpu_memory_list)
#         max_memory = max(gpu_memory_list)
#         avg_memory = sum(gpu_memory_list) / len(gpu_memory_list)

#         print("\nGPU memory usage")
#         print(f"Minimum GPU memory usage: {min_memory:.2f} MB")
#         print(f"Maximum GPU memory usage: {max_memory:.2f} MB")
#         print(f"Average GPU memory usage: {avg_memory:.2f} MB\n")

# def cuda_memory(exit_flag):
#     gpu_memory_list = []

#     while not exit_flag.is_set():
#         try:
#             cuda_memory_kb = torch.cuda.memory_allocated()
#             current_memory_mb = cuda_memory_kb / (1024 ** 2)
            
#             gpu_memory_list.append(current_memory_mb)

#             time.sleep(0.5)
#         except Exception as e:
#             print(f"Error while monitoring GPU memory usage: {e}")

#     time.sleep(1)
#     if gpu_memory_list:
#         min_memory = min(gpu_memory_list)
#         max_memory = max(gpu_memory_list)
#         avg_memory = sum(gpu_memory_list) / len(gpu_memory_list)

#         print("\nGPU memory usage(CUDA)")
#         print(f"Minimum GPU memory usage: {min_memory:.2f} MB")
#         if max_memory < 1000:
#             print(f"Maximum GPU memory usage: {max_memory:.2f} MB")
#         else:
#             print(f"Maximum GPU memory usage: {max_memory/1024:.2f} GB")
#         print(f"Average GPU memory usage: {avg_memory:.2f} MB\n")

# def main(c):
#     exit_flag = threading.Event()

#     gpu_percentage_thread = threading.Thread(
#         target=gpu_percentage,
#         args=(exit_flag,int(c.gpu),)
#     )
#     gpu_percentage_thread.start()

#     gpu_usage_thread = threading.Thread(
#         target=gpu_memory,
#         args=(exit_flag,int(c.gpu),)
#     )
#     gpu_usage_thread.start()

#     cuda_usage_thread = threading.Thread(
#         target=cuda_memory,
#         args=(exit_flag,)
#     )
#     cuda_usage_thread.start()


#     c = parsing_args(c)
#     os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
#     init_seeds(seed=c.seed)
#     c.version_name = 'msflow_{}_{}pool_pl{}'.format(c.extractor, c.pool_type, "".join([str(x) for x in c.parallel_blocks]))
#     c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.class_name)

#     # inspect_args(c.mode)

#     if c.mode == 'use':
#         from inference import model_usage
#         model_usage(c)
#         return
#     print(f'extractor = {c.extractor}')
#     train(c)


#     exit_flag.set()

#     gpu_percentage_thread.join()
#     gpu_usage_thread.join()
#     cuda_usage_thread.join()

# if __name__ == '__main__':
#     import default as c

#     main(c)



def gpu_percentage(exit_flag, gpu, result_queue):
    gpu_usage_list = []

    while not exit_flag.is_set():
        try:
            gpu_usage = GPUtil.getGPUs()[gpu].load * 100.0
            gpu_usage_list.append(gpu_usage)

            time.sleep(0.5)
        except Exception as e:
            print(f"Error while monitoring GPU usage: {e}")

    if gpu_usage_list:
        min_usage = min(gpu_usage_list)
        max_usage = max(gpu_usage_list)
        avg_usage = sum(gpu_usage_list) / len(gpu_usage_list)

        result_queue.put({"min_usage": min_usage, "max_usage": max_usage, "avg_usage": avg_usage})

def gpu_memory(exit_flag, gpu, result_queue):
    gpu_memory_list = []

    while not exit_flag.is_set():
        try:
            gpu_memory_kb = GPUtil.getGPUs()[gpu].memoryUsed
            gpu_memory_mb = gpu_memory_kb / 1024  # KB를 MB로 변환 -> 근데 크기가 크면 자동으로 단위 변경해줘서 GB로 나옴
            gpu_memory_list.append(gpu_memory_mb)

            time.sleep(0.5)
        except Exception as e:
            print(f"Error while monitoring GPU memory usage: {e}")

    time.sleep(0.4)
    if gpu_memory_list:
        min_memory = min(gpu_memory_list)
        max_memory = max(gpu_memory_list)
        avg_memory = sum(gpu_memory_list) / len(gpu_memory_list)

        result_queue.put({"min_memory": min_memory, "max_memory": max_memory, "avg_memory": avg_memory})

def cuda_memory(exit_flag, result_queue):
    gpu_memory_list = []

    while not exit_flag.is_set():
        try:
            cuda_memory_kb = torch.cuda.memory_allocated()
            current_memory_mb = cuda_memory_kb / (1024 ** 2)
            
            gpu_memory_list.append(current_memory_mb)

            time.sleep(0.5)
        except Exception as e:
            print(f"Error while monitoring GPU memory usage: {e}")

    time.sleep(1)
    if gpu_memory_list:
        min_memory = min(gpu_memory_list)
        max_memory = max(gpu_memory_list)
        avg_memory = sum(gpu_memory_list) / len(gpu_memory_list)

        result_queue.put({"cuda_min_memory": min_memory, "cuda_max_memory": max_memory, "cuda_avg_memory": avg_memory})

def main(c):
    exit_flag = threading.Event()
    result_queue = queue.Queue()  # 결과를 저장할 큐 추가

    gpu_percentage_thread = threading.Thread(
        target=gpu_percentage,
        args=(exit_flag, int(c.gpu), result_queue)
    )
    gpu_percentage_thread.start()

    gpu_usage_thread = threading.Thread(
        target=gpu_memory,
        args=(exit_flag, int(c.gpu), result_queue)
    )
    gpu_usage_thread.start()

    cuda_usage_thread = threading.Thread(
        target=cuda_memory,
        args=(exit_flag, result_queue)
    )
    cuda_usage_thread.start()

    c = parsing_args(c)
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    init_seeds(seed=c.seed)
    c.version_name = 'msflow_{}_{}pool_pl{}'.format(c.extractor, c.pool_type, "".join([str(x) for x in c.parallel_blocks]))
    c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.class_name)

    if c.mode == 'use':
        from inference import model_usage
        model_usage(c)
        return

    print(f'extractor = {c.extractor}')
    train(c)
    
    exit_flag.set()

    gpu_percentage_thread.join()
    gpu_usage_thread.join()
    cuda_usage_thread.join()

    # 결과 큐에서 값을 가져와서 반환
    gpu_info = {"gpu_percentage": None, "gpu_memory": None, "cuda_memory": None}

    while not result_queue.empty():
        result = result_queue.get()
        if "avg_usage" in result:
            gpu_info["gpu_percentage"] = result
        elif "avg_memory" in result:
            gpu_info["gpu_memory"] = result
        elif "cuda_avg_memory" in result:
            gpu_info["cuda_memory"] = result

    gpu_info['gpu_num'] = c.gpu
    return gpu_info

if __name__ == '__main__':
    import default as c

    gpu_info = main(c)
    print(f"\n#GPU{gpu_info['gpu_num']} Information:")
    print("#GPU Percentage:", gpu_info["gpu_percentage"])
    print("#GPU Memory:", gpu_info["gpu_memory"])
    print("#CUDA Memory:", gpu_info["cuda_memory"])