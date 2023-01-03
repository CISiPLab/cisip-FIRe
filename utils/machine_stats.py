import logging
import os
import subprocess
import sys

import psutil
import torch


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def print_stats():
    is_windows = sys.platform.startswith('win')
    if not is_windows:
        cpu_name = (subprocess.check_output('lscpu | grep \"Model name\"',
                                            shell=True).strip()).decode().split(':')[1].strip()
        memory = psutil.virtual_memory()
        memory_total = memory.total / (1024 * 1024 * 1024)
        memory_avail = memory.available / (1024 * 1024 * 1024)
        gpu_memory = get_gpu_memory()

        lines = [
            f'GPU counts: {torch.cuda.device_count()}',
            f'GPU 0: {torch.cuda.get_device_properties(0)}',
            f'GPU Free Mem: {gpu_memory[0]}MB',
            f'CPU cores: {os.cpu_count()}',
            f'CPU name: {cpu_name}',
            f'RAM: {memory_total - memory_avail:.2f}GB/{memory_total:.2f}GB',
        ]

        for line in lines:
            logging.info(line)
    else:
        print('Windows do not have "lscpu"')
