from __future__ import print_function
import argparse
import timeit
from cvxpy import mixed_norm
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import sys
import numpy as np
import os
import pandas as pd
import torchvision
import time
import torch.multiprocessing as mp
sys.path.append('/home/mzhang/work/ASPLOS23/collect_metric/')

from torch.nn import DataParallel
from multiprocessing import Process, Manager, Value
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from models import *

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--warmup_iter', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=50, help='number of training benchmark epochs')
parser.add_argument('--data_dir', type=str, default="~/data/", help='Data directory')
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')
parser.add_argument("--world_size", type=int)
parser.add_argument("--node_rank", type=int)
parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Total time to run the code')
parser.add_argument('--master_port', type=str, default='47020', help='Total time to run the code')

args = parser.parse_args()


def benchmark_cifar_ddp(local_rank, node_rank, local_size, world_size, model_name, batch_size, mixed_precision):
    t_start = time.time()
    # initialize the process group
    rank = local_rank + node_rank * local_size
    dist.init_process_group(backend="nccl", 
                            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            rank=rank, 
                            world_size=world_size)
    # this function is responsible for synchronizing and successfully communicate across multiple process
    # involving multiple GPUs.

    print(f"Running Distributed ResNet rank {rank} on node {node_rank}.")
    torch.manual_seed(0)

    # Model
    # print('==> Building model..')
    if model_name == 'VGG':
        model = VGG('VGG11')
    elif model_name == 'ShuffleNetV2': 
        model = ShuffleNetV2(net_size=0.5)
    else:
        model = eval(model_name)()
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # Dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, num_workers=2, sampler=train_sampler)
    # data, target = next(iter(trainloader))
    # data, target = data.cuda(), target.cuda()

    # Train
    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    iter_num = 0
    exit_flag = False
    model.train()
    # Prevent total batch number < warmup+benchmark situation
    while True:
        for inputs, targets in trainloader:
            # Warm-up: previous 10 iters
            if iter_num == args.warmup_iter-1:
                t_warmend = time.time()
            # Reach timeout: exit benchmark
            if time.time() - t_start >= args.total_time:
                t_end = time.time()
                t_pass = t_end - t_warmend
                exit_flag = True
                break
            optimizer.zero_grad()
            if mixed_precision:
                inputs, targets = inputs.to(local_rank), targets.to(local_rank)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                inputs, targets = inputs.to(local_rank), targets.to(local_rank)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            iter_num += 1
        if exit_flag:
            break

    img_sec = local_size * (iter_num - args.warmup_iter) * batch_size / t_pass
    print(img_sec)


if __name__ == '__main__':
    model_name = 'ResNet18'
    batch_size = 32
    mixed_precision = 0
    local_size = torch.cuda.device_count()

    mp.spawn(benchmark_cifar_ddp, args=(args.node_rank, local_size, args.world_size, model_name, batch_size, mixed_precision, ), nprocs=local_size, join=True)
