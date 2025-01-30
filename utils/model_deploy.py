# Description: This file contains some utility functions.

import time

import torch
import torch.nn as nn


device_cpu = torch.device('cpu')
torch.set_num_threads(1)

device_gpu = torch.device('cuda')

def measure_inference_latency(net, device):

    random_inputs = torch.randn(100, 3, 64, 64).to(device_gpu)

    net = net.to(device_gpu)

    time_list = []
    for i in range(50):

        net.eval()
        total_time = 0.0
        num_batches = 0

        with torch.no_grad():
            # Measure time for one forward pass
            start_time = time.time()
            outputs = net(random_inputs)
            end_time = time.time()

            total_time += (end_time - start_time)
            num_batches += 1

        avg_latency = (total_time / num_batches) * 1000  # convert to milliseconds
        # print(f'Average latency per batch: {avg_latency:.2f} ms')

        time_list.append(avg_latency)
    
    # print(f'Average latency per batch over 50 runs: {sum(time_list) / len(time_list):.2f} ms')
    return sum(time_list) / len(time_list)