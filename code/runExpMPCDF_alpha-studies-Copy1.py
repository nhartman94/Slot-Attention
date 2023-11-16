'''
Steering script for submitting trainings for 
initial optimizations

Nicole Hartman 
Summer 2023
'''

import os

def writeSlurmFile(pythonCmd, job_name, useGPU=True):
    """
    Write a LSF job file to submit a python comand with a specified jobName
    """
    # Open the file
    f = open(f"{SLURM_DIR}/{job_name}.sh", "w")
    f.write("#!/bin/bash -l\n")

    options = {
        "ntasks": 1,
        "constraint": "gpu",
        "gres": "gpu:a100:1",
        "mem": 125000,
        "cpus-per-task": 4,
        "time": "12:00:00",
        "job-name": job_name,
        "output": f"log_files/{job_name}.log",
        "error": f"log_files/{job_name}.log",
        #"mail-type": "ALL",
        #"mail-user": "nicole.hartman@tum.de",
    }

    for k, v in options.items():
        f.write("#SBATCH --{} {}\n".format(k, v))
        
    print('Preparing to run on the MPCDF (TUM) machines')
    
    f.write("module purge\n")

    # Create and load conda environment
    f.write("module load anaconda/3/2021.11 # <-> python 3.9.\n")
    f.write("module load cuda/11.6\n")
    f.write("module load cudnn/8.8\n")
    f.write("module load pytorch/gpu-cuda-11.6/2.0.0\n")
    
    f.write("conda env create -f gpu_env.yaml\n")
    f.write("conda activate gpu_env\n")
    
    f.write("pwd\n")
    
    f.write(pythonCmd+"\n")    
    f.close()

SLURM_DIR = "slurm_scripts"

if not os.path.exists(SLURM_DIR):
    os.mkdir(SLURM_DIR)

# NH: 22.10.23
# ISA model with L_tot = L_bce + alpha + L_mse
for i in range(16):

    cID = f'isa-alpha3_' + str(i)
    cmd = f"python train.py --config configs/{cID}.yaml --warm_start"
    cmd += " --warm_start_config configs/isa-cosine-decay.yaml --device cuda:0 "
    writeSlurmFile(cmd, cID, useGPU=True)
    os.system(f"sbatch {SLURM_DIR}/{cID}.sh")
