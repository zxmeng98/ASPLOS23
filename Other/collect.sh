CUDA_VISIBLE_DEVICES=0 python dp_benchmark.py --model mobilenet_v3_small  & 
CUDA_VISIBLE_DEVICES=0 python dp_benchmark.py --model mobilenet_v3_small &
python gpu_meter.py 

#2>&1 | tee a.log  

sudo ./mps_scripts/init_mps_for_all_gpus.sh
sudo ./mps_scripts/set_mps_env_for_all_gpus.sh 

sudo ./mps_scripts/stop_mps_for_all_gpus.sh

nvidia-cuda-mps-control -d 

CUDA_VISIBLE_DEVICES=0 python profile.py 


CUDA_VISIBLE_DEVICES=0 python profile_cifar.py & 
CUDA_VISIBLE_DEVICES=0 python profile_cifar.py &
python gpu_meter.py 


python profile_cifar_ddp.py & python profile_cifar_ddp.py --master_port 12335

python profile_imagenet_ddp.py & python profile_imagenet_ddp.py --master_port 12335