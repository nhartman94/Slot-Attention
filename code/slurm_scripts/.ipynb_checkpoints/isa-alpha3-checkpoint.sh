#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --constraint gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem 125000
#SBATCH --cpus-per-task 4
#SBATCH --time 12:00:00
#SBATCH --job-name isa-alpha3
#SBATCH --output log_files/isa-alpha3.log
#SBATCH --error log_files/isa-alpha3.log
module purge
conda create --name myenv python=3.9.7
conda activate myenv
pwd
pip install -r requirements1.txt

pwd
python train.py --config configs/isa-alpha3.yaml --warm_start --warm_start_config configs/isa-cosine-decay.yaml --device cuda:0 
