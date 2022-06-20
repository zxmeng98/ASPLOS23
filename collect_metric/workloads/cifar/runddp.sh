CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 profile_cifar_ddp.py

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 profile_cifar_ddp.py