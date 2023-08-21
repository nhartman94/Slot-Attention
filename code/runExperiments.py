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
        "account": "atlas",
        "partition": "atlas",
        "ntasks": 1,
        "cpus-per-task": 1,
        "mem-per-cpu": "4g",
        "time": "24:00:00",  # hour: minute: second
        # Specify the job name, and input and output logfiles
        "job-name": job_name,
        "output": f"log_files/{job_name}.log",
        "error": f"log_files/{job_name}.log",
    }

    for k, v in options.items():
        f.write("#SBATCH --{} {}\n".format(k, v))

    if useGPU:
        f.write("#SBATCH --gpus=1\n")

    # This command spins up the singularity image to allow the training to use the gpus
    f.write(
        "export SINGULARITY_IMAGE_PATH=/sdf/group/ml/software/images/slac-ml/20211101.0/slac-ml@20211101.0.sif\n"
    )

    # cd into the right directory
    f.write("cd /gpfs/slac/atlas/fs1/d/nhartman/Slot\ Attention/code\n")
    f.write(
        "singularity exec --nv -B /sdf,/gpfs ${SINGULARITY_IMAGE_PATH} "
        + pythonCmd
    )

    # Close the file
    f.close()

SLURM_DIR = "slurm_scripts"

'''
#Checking varying the batchsize


cID = '2blobs-bs-256'

cmd = f"python train.py --config configs/{cID}.yaml --device cuda:0"
job = f"{cID}"
writeSlurmFile(cmd, job, useGPU=True)
os.system(f"sbatch {SLURM_DIR}/{job}.sh")
'''

# Test the temperature
#for ctag in ['muP-D','LHT','sqrtD']:
#for ctag in ['sqrtD']: # muP-D
#    cID = f'2rings-{ctag}'
#    cmd = f"python train.py --config configs/{cID}.yaml --device cuda:0"
#    cmd += " --warm_start "
#    writeSlurmFile(cmd, cID, useGPU=True)
#    os.system(f"sbatch {SLURM_DIR}/{cID}.sh")
#
## Suggestion from Steffan: Decrease the query dim of SA
#for Q in [4,8]:
#    cID = f'2rings-sqrtD-Q{Q}'
#    cmd = f"python train.py --config configs/{cID}.yaml --device cuda:0"
#    cmd += " --warm_start "
#    writeSlurmFile(cmd, cID, useGPU=True)
#    os.system(f"sbatch {SLURM_DIR}/{cID}.sh")

# 21.08.23 smaller Q dim didn't work as well, so let's go the other direction!
#for Q in [16,32,64,128,256]:
#for Q in [16,32,64]:
#    cID = f'2rings-sqrtD-Q{Q}'
#    cmd = f"python train.py --config configs/{cID}-10k-decay.yaml --device cuda:0"
#    cmd += f" --warm_start --warm_start_config configs/{cID}.yaml"
#    writeSlurmFile(cmd, f"{cID}-10k-decay", useGPU=True)
#    os.system(f"sbatch {SLURM_DIR}/{cID}-10k-decay.sh")

# Try increasing the # iters you're solving the problem in
for T in [2,3,4,5]:
    cID = f'2rings-T{T}'
    cmd = f"python train.py --config configs/{cID}.yaml --device cuda:0"
    cmd += f" --warm_start --warm_start_config configs/2rings-sqrtD.yaml"
    writeSlurmFile(cmd, cID, useGPU=True)
    os.system(f"sbatch {SLURM_DIR}/{cID}.sh")


