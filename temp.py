import time
import GPUtil
import threading

def gpu_percentage(exit_flag, gpu):
    gpu_usage_list = []

    while not exit_flag.is_set():
        try:
            gpu_usage = GPUtil.getGPUs()[gpu].load * 100.0
            gpu_usage_list.append(gpu_usage)

            time.sleep(0.5)
        except Exception as e:
            print(f"Error while monitoring GPU usage: {e}")

    if gpu == 0:
        time.sleep(1)
    else:
        time.sleep(3)

    if gpu_usage_list:
        min_usage = min(gpu_usage_list)
        max_usage = max(gpu_usage_list)
        avg_usage = sum(gpu_usage_list) / len(gpu_usage_list)

        print(f"\nGPU{gpu} percentage usage")
        print(f"Minimum GPU usage: {min_usage:.2f}%")
        print(f"Maximum GPU usage: {max_usage:.2f}%")
        print(f"Average GPU usage: {avg_usage:.2f}%\n")

def gpu_memory(exit_flag, gpu):
    gpu_memory_list = []

    while not exit_flag.is_set():
        try:
            gpu_memory_kb = GPUtil.getGPUs()[gpu].memoryUsed
            gpu_memory_mb = gpu_memory_kb / 1024  # KB를 MB로 변환
            gpu_memory_list.append(gpu_memory_mb)

            time.sleep(0.5)
        except Exception as e:
            print(f"Error while monitoring GPU memory usage: {e}")

    if gpu == 0:
        time.sleep(2)
    else:
        time.sleep(4)

    if gpu_memory_list:
        min_memory = min(gpu_memory_list)
        max_memory = max(gpu_memory_list)
        avg_memory = sum(gpu_memory_list) / len(gpu_memory_list)

        print(f"\nGPU{gpu} memory usage")
        print(f"Minimum GPU memory usage: {min_memory:.2f} MB")
        print(f"Maximum GPU memory usage: {max_memory:.2f} MB")
        print(f"Average GPU memory usage: {avg_memory:.2f} MB\n")


if __name__ == '__main__':
    exit_flag = threading.Event()

    gpu0_percentage_thread = threading.Thread(
        target=gpu_percentage,
        args=(exit_flag,0,)
    )
    gpu0_percentage_thread.start()

    gpu0_usage_thread = threading.Thread(
        target=gpu_memory,
        args=(exit_flag,0,)
    )
    gpu0_usage_thread.start()

    gpu1_percentage_thread = threading.Thread(
        target=gpu_percentage,
        args=(exit_flag,1,)
    )
    gpu1_percentage_thread.start()

    gpu1_usage_thread = threading.Thread(
        target=gpu_memory,
        args=(exit_flag,1,)
    )
    gpu1_usage_thread.start()


    time.sleep(60)


    exit_flag.set()

    gpu0_percentage_thread.join()
    gpu0_usage_thread.join()


    gpu1_percentage_thread.join()
    gpu1_usage_thread.join()
