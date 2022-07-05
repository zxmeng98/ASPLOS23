from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import sys
import os
import torchvision
import torch.multiprocessing as mp
sys.path.append('/home/mzhang/work/ASPLOS23/collect_metric/')

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
parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Total time to run the code')
parser.add_argument('--master_port', type=str, default='47020', help='Total time to run the code')
parser.add_argument('--n_epochs', type=int, default=2, help='Total number of epochs for training')

args = parser.parse_args()


# ------ Setting up the distributed environment -------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successfully communicate across multiple process
    # involving multiple GPUs.


def cleanup():
    dist.destroy_process_group()


def benchmark_cifar_ddp(rank, model_name, batch_size, mixed_precision, gpu_id):
    print(f"Running Distributed ResNet on rank {rank}.")
    setup(rank, len(gpu_id))
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    # Model
    # print('==> Building model..')
    if model_name == 'VGG':
        model = VGG('VGG11')
    elif model_name == 'ShuffleNetV2': 
        model = ShuffleNetV2(net_size=0.5)
    else:
        model = eval(model_name)()
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss().to(rank)
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

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,  rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, num_workers=2, sampler=train_sampler)
    # data, target = next(iter(trainloader))
    # data, target = data.cuda(), target.cuda()

    # Train
    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    exit_flag = False
    model.train()
    # Prevent total batch number < warmup+benchmark situation
    for epoch in range(args.n_epochs):
        iter_num = 0
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            if mixed_precision:
                inputs, targets = inputs.to(rank), targets.to(rank)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                inputs, targets = inputs.to(rank), targets.to(rank)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            iter_num += 1
        if epoch == 1: 
            print('End warmup')

    img_sec = len(gpu_id) * iter_num * (args.n_epochs-2) * batch_size
    print(f'End master port: {args.master_port}, images: {img_sec}')
    
    cleanup()

if __name__ == '__main__':
    model_name = 'EfficientNetB0'
    batch_size = 64
    mixed_precision = 0
    gpu_id = [0,1,2,3]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)
    # world_size = 4

    mp.spawn(benchmark_cifar_ddp, args=(model_name, batch_size, mixed_precision, gpu_id, ), nprocs=len(gpu_id), join=True)
