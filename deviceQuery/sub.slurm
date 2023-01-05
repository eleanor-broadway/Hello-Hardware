#!/bin/bash
#
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=z04

# Load the required modules
module load nvidia/nvhpc

srun ./deviceQuery
