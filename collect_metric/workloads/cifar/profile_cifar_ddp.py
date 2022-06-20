from __future__ import print_function
import argparse
import timeit
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
from multiprocessing import Process, Manager, Value

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from models import *


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
parser.add_argument("--gpu", default=1, type=int, help="GPU id to use. Only work when use single gpu.")
parser.add_argument(
    "--num-warmup-batches", type=int, default=1, help='number of warm-up batches that don"t count towards benchmark'
)
parser.add_argument("--num-batches-per-iter", type=int, default=1, help="number of batches per benchmark iteration")
parser.add_argument("--num-iters", type=int, default=1, help="number of benchmark iterations")
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--warmup_epoch', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=50, help='number of training benchmark epochs')
parser.add_argument('--data_dir', type=str, default="~/data/", help='Data directory')
parser.add_argument('--total_time', type=int, default=20, help='Total time to run the code')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

args = parser.parse_args()
print(args.local_rank)

def benchmark_cifar(model_name, batch_size, mixed_precision, gpu_id):
    t_start = time.time()
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    # set the device
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # Model
    # print('==> Building model..')
    if model_name == 'VGG':
        model = VGG('VGG11')
    elif model_name == 'ShuffleNetV2': 
        model = ShuffleNetV2(net_size=0.5)
    else:
        model = eval(model_name)()
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
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
    if len(gpu_id) > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,  rank=args.local_rank)
    else:
        train_sampler = None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, num_workers=2, sampler=train_sampler)
    # data, target = next(iter(trainloader))
    # data, target = data.cuda(), target.cuda()

    # Train
    def benchmark_step():
        iter_num = 0
        exit_flag = False
        model.train()
        # Prevent total batch number < warmup+benchmark situation
        while True:
            for inputs, targets in trainloader:
                # Warm-up: previous 10 iters
                if iter_num == args.warmup_epoch-1:
                    t_warmend = time.time()
                # Reach timeout: exit benchmark
                if iter_num == args.benchmark_epoch + args.warmup_epoch -1:
                    t_end = time.time()
                    t_pass = t_end - t_warmend
                    exit_flag = True
                    break
                optimizer.zero_grad()
                if mixed_precision:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                iter_num += 1
            if exit_flag:
                break
        return t_pass, iter_num

    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass, iter_num = benchmark_step()
    img_sec = len(gpu_id) * args.benchmark_epoch * batch_size / t_pass
  
    # Results
    return img_sec


speed = benchmark_cifar('ResNet18', 32, 0, [0, 1])
print(speed)