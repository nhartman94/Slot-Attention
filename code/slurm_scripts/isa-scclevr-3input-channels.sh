#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --constraint gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem 125000
#SBATCH --cpus-per-task 4
#SBATCH --time 12:00:00
#SBATCH --job-name isa-scclevr-3input-channels
#SBATCH --output log_files/isa-scclevr-3input-channels.log
#SBATCH --error log_files/isa-scclevr-3input-channels.log
module purge
module load anaconda/3/2021.11 # <-> python 3.9.
module load cuda/11.6
module load cudnn/8.8
module load pytorch/gpu-cuda-11.6/2.0.0
conda activate gpu_env
pwd
python train-scclevr-encoder-study.py --config configs/isa-scclevr-3input-channels.yaml --warm_start --warm_start_config configs/isa-cosine-decay.yaml --device cuda:0 
