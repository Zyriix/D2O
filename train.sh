#!/bin/bash

export TORCH_CUDNN_V8_API_ENABLED=1

# CHANGE nproc_per_node to your desired number of GPUs

# D2O
# CIFAR-10 uncond
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=0 --backbone=1 --ema=0.5  --dropout=0  --batch=256 --dlr=1e-4 --lr=1e-4 --cond=0 --arch=ncsnpp --duration=5 \
        --outdir=training-runs-uncond-cifar10-d2o --transfer=pretrained/edm-cifar10-32x32-uncond-ve.pkl \
        --data=datasets/cifar10-32x32.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1' \
        --augment=0 --ada-target=0 --use-gp=True --d-type='xl'  --seed=42 \
        --precond=d2o --loss=d2o

# CIFAR-10 cond
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=0 --backbone=1 --ema=0.5  --dropout=0  --batch=256 --dlr=1e-4 --lr=1e-4 --cond=1 --arch=ncsnpp --duration=5   \
        --outdir=training-runs-cond-cifar10-d2o --transfer=pretrained/edm-cifar10-32x32-cond-ve.pkl \
        --data=datasets/cifar10-32x32.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1'   \
        --augment=0 --ada-target=0 --use-gp=True --d-type='xl'  --seed=42  \
        --precond=d2o --loss=d2o
# AFHQv2
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=0 --backbone=0 --ema=0.5  --dropout=0  --batch=256 --dlr=4e-5 --lr=2e-5 --cond=0 --arch=ncsnpp --duration=10 --cres=1,2,2,2  \
        --outdir=training-runs-uncond-afhqv2-d2o  --transfer=pretrained/edm-afhqv2-64x64-uncond-ve.pkl \
        --data=datasets/afhqv2-64x64.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1' \
        --augment=0 --ada-target=0 --use-gp=True --d-type='xl'  --seed=42  \
        --precond=d2o --loss=d2o

# FFHQ
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=0 --backbone=0 --ema=0.5  --dropout=0  --batch=256 --dlr=4e-5 --lr=2e-5 --cond=0 --arch=ncsnpp --duration=10 --cres=1,2,2,2   \
        --outdir=training-runs-uncond-ffhq-d2o  --transfer=pretrained/edm-ffhq-64x64-uncond-ve.pkl \
        --data=datasets/ffhq-64x64.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1' \
        --augment=0 --ada-target=0 --use-gp=True --d-type='xl'  --seed=42  \
        --precond=d2o --loss=d2o



# ImageNet
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=0 --backbone=0 --ema=50  --dropout=0  --batch=512 --dlr=4e-5 --lr=8e-6  --cond=1 --arch=adm --duration=10  \
        --outdir=training-runs-cond-imagenet-d2o --transfer=pretrained/edm-imagenet-64x64-cond-adm.pkl \
        --data=datasets/imagenet-64x64.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1' \
        --augment=0 --ada-target=0 --use-gp=True --d-type='xl'  --seed=42  \
        --precond=d2o --loss=d2o


# D2O-F
# CIFAR-10 uncond
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=1 --backbone=1 --ema=0.5  --dropout=0  --batch=256 --dlr=1e-4 --lr=1e-4 --cond=0 --arch=ncsnpp --duration=5 \
        --outdir=training-runs-uncond-cifar10-d2o-f --transfer=pretrained/edm-cifar10-32x32-uncond-ve.pkl \
        --data=datasets/cifar10-32x32.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1' \
        --ada-target=0 --use-gp=True --d-type='xl'  --seed=42 \
        --precond=d2o --loss=d2o

# CIFAR-10 cond
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=1 --backbone=1 --ema=0.5  --dropout=0  --batch=256 --dlr=1e-4 --lr=1e-4 --cond=1 --arch=ncsnpp --duration=5   \
        --outdir=training-runs-cond-cifar10-d2o-f --transfer=pretrained/edm-cifar10-32x32-cond-ve.pkl \
        --data=datasets/cifar10-32x32.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1'   \
        --ada-target=0 --use-gp=True --d-type='xl'  --seed=42  \
        --precond=d2o --loss=d2o
# AFHQv2
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=1 --backbone=0 --ema=0.5  --dropout=0  --batch=256 --dlr=4e-5 --lr=2e-5 --cond=0 --arch=ncsnpp --duration=10 --cres=1,2,2,2  \
        --outdir=training-runs-uncond-afhqv2-d2o-f --transfer=pretrained/edm-afhqv2-64x64-uncond-ve.pkl  \
        --data=datasets/afhqv2-64x64.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1' \
        --ada-target=0 --use-gp=True --d-type='xl'  --seed=42  \
        --precond=d2o --loss=d2o

# FFHQ
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=1 --backbone=0 --ema=0.5  --dropout=0  --batch=256 --dlr=4e-5 --lr=2e-5 --cond=0 --arch=ncsnpp --duration=10 --cres=1,2,2,2  \
        --outdir=training-runs-uncond-ffhq-d2o-f --transfer=pretrained/edm-ffhq-64x64-uncond-ve.pkl \
        --data=datasets/ffhq-64x64.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1' \
        --ada-target=0 --use-gp=True --d-type='xl'  --seed=42  \
        --precond=d2o --loss=d2o



# ImageNet
torchrun --standalone --nproc_per_node=4 train.py --workers=4 \
        --freeze=1 --backbone=0 --ema=50  --dropout=0  --batch=512 --dlr=4e-5 --lr=8e-6  --cond=1 --arch=adm --duration=10  \
        --outdir=training-runs-cond-imagenet-d2o-f --transfer=pretrained/edm-imagenet-64x64-cond-adm.pkl \
        --data=datasets/imagenet-64x64.zip \
        --fp16=True --loss-type='ns'  --r1-gamma=1e-4 \
        --diffaug=False --r1-type='r1' \
        --ada-target=0 --use-gp=True --d-type='xl'  --seed=42  \
        --precond=d2o --loss=d2o
