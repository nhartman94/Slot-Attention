#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --constraint gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem 125000
#SBATCH --cpus-per-task 4
#SBATCH --time 12:00:00
#SBATCH --job-name isa-alpha3_11
#SBATCH --output log_files/isa-alpha3_11.log
#SBATCH --error log_files/isa-alpha3_11.log
module purge
module load anaconda/3/2021.11 # <-> python 3.9.
module load cuda/11.6
module load cudnn/8.8
module load pytorch/gpu-cuda-11.6/2.0.0
conda env create -f gpu_env.yaml
conda activate gpu_env
pwd
python train.py --config configs/isa-alpha3_11.yaml --warm_start --warm_start_config configs/isa-cosine-decay.yaml --device cuda:0 
