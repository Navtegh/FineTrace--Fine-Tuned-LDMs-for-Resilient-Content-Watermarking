#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode33
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=training_output_atck_nores_2_sch.out
#SBATCH --error=training_output2_atck_nores_2_sch.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun torchrun --nproc_per_node=1 main.py \
  --val_dir /w/284/navtegh/stable_signature-main/hidden/val2014/ --train_dir /w/284/navtegh/stable_signature-main/hidden/train2014 --output_dir output_atck_nores2 --eval_freq 5 \
  --img_size 256 --num_bits 32  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 1.0 --p_blur 0.0 --p_rot 1.0 --p_crop 1.0 --p_res 0.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels True --attenuation none \
  --loss_w_type mse --loss_margin 1 --dist False


