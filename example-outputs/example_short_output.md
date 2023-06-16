# Truncated output of hello-hardware.cu code.

This prints concise information about the hardware available to a program given the parameters defined in the submission script.

- The "Node X has X active GPUs" prints once for each rank. This is because CUDA calls only queries the local hardware for CUDA-capable devices so only reports the number of GPUs per compute node.


```
!/bin/bash
#SBATCH --time=00:20:00

#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=slurm-%j-GPU2.out

module load nvidia/nvhpc-nompi
module load mpt

export OMP_NUM_THREADS=4

cat $0

time srun --ntasks=8 --cpus-per-task=4 --hint=nomultithread ./a.out

Detecting 8 ranks with 4 OMP threads per rank
Node r2i6n4 has 4 active GPUs
Node r2i6n3 has 4 active GPUs
Node r2i6n4 has 4 active GPUs
Node r2i6n3 has 4 active GPUs
Node r2i6n4 has 4 active GPUs
Node r2i6n3 has 4 active GPUs
Node r2i6n4 has 4 active GPUs
Node r2i6n3 has 4 active GPUs
```
