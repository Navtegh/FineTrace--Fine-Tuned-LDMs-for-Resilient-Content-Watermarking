#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode23
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=training_output_sch.out
#SBATCH --error=training_output2_sch.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 finetune_ldm_decoder.py --num_keys 1 \
    --batch_size 1 \
    --img_size 256\
    --ldm_config /w/284/navtegh/stable_signature-main/v2-inference.yaml \
    --ldm_ckpt /w/284/navtegh/stable_signature-main/v2-1_512-ema-pruned.ckpt \
    --msg_decoder_path /w/284/navtegh/stable_signature-main/hidden/output_atck_nores/checkpoint120.pth \
    --train_dir /w/284/navtegh/stable_signature-main/hidden/train2014 \
    --val_dir /w/284/navtegh/stable_signature-main/valid



