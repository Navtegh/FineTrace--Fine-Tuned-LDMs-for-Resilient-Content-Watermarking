#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode23
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=training_output_sc.out
#SBATCH --error=training_output2_sc.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun torchrun --nproc_per_node=1 test.py 



