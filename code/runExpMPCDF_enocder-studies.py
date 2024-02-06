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
    
    #f.write("conda env create -f gpu_env.yaml\n")
    f.write("conda activate gpu_env\n")
    #f.write("conda env update -f gpu_env.yaml\n")
    
    f.write("pwd\n")
    
    f.write(pythonCmd+"\n")    
    f.close()

SLURM_DIR = "slurm_scripts"

if not os.path.exists(SLURM_DIR):
    os.mkdir(SLURM_DIR)

# SA: 06.01.24
# train on scCLEVR dataset
# encoder study

# SA: 02.02.2024: train longer and with no warm up as warm up is with dense rings

#cID = f'isa-scclevr-EncStudy-moreCNN'
#cID = f'isa-scclevr-EncStudy-myResNet'
#for cID in ['isa-scclevr-EncStudy-ResNet_S1', 'isa-scclevr-EncStudy-ResNet_S2']:
#cID = 'isa-scclevr-EncStudy-ResNet_S1'
#cID = f'isa-scclevr-EncStudy-moreCNN20'

cID = f'isa-scclevr-EncStudy-myBigResNet-longTraining'
cmd = f"python train-scclevr-encoder-study.py --config configs/{cID}.yaml "#--warm_start"
#cmd += " --warm_start_config configs/isa-cosine-decay.yaml --device cuda:0 "
writeSlurmFile(cmd, cID, useGPU=True)
os.system(f"sbatch {SLURM_DIR}/{cID}.sh")
