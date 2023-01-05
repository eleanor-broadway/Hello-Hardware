# Full output of hello-hardware.cu code.

Prints:
- Hello from rank myrank/world
- Hello from thread mythread/world from rank myrank/world
- My node ID has N active GPUs
- Additional GPU info
- Total summary of threads, ranks and GPUs.

The output is messy but correctly reports the information requested in the submission script.

```
#!/bin/bash
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

Hello from rank 0/8
Hello from rank 5/8
Hello from rank 6/8
Hello from rank 7/8
Hello from rank 4/8
Hello from thread 0/4 from rank 7/8
Hello from thread 2/4 from rank 7/8
Hello from thread 3/4 from rank 7/8
Hello from thread 1/4 from rank 7/8
Hello from thread 0/4 from rank 4/8
Hello from thread 2/4 from rank 4/8
Hello from thread 3/4 from rank 4/8
Hello from thread 1/4 from rank 4/8
Hello from thread 1/4 from rank 6/8
Hello from thread 2/4 from rank 6/8
Hello from thread 3/4 from rank 6/8
Hello from thread 0/4 from rank 6/8
Hello from thread 0/4 from rank 5/8
Hello from thread 1/4 from rank 5/8
Hello from thread 2/4 from rank 5/8
Hello from thread 3/4 from rank 5/8
Node r2i6n4 has 4 active GPUs
Node r2i6n4 has 4 active GPUs
Node r2i6n4 has 4 active GPUs
Node r2i6n4 has 4 active GPUs
Hello from Tesla V100-SXM2-16GB GPU 0 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 0 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 0 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 0 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 1 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 1 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 1 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 1 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 2 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 2 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 2 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 2 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 3 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 3 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 3 / 4 on node r2i6n4
Hello from Tesla V100-SXM2-16GB GPU 3 / 4 on node r2i6n4
Hello from rank 1/8
Hello from rank 2/8
Hello from rank 3/8
Hello from thread 3/4 from rank 3/8
Hello from thread 2/4 from rank 3/8
Hello from thread 1/4 from rank 3/8
Hello from thread 0/4 from rank 3/8
Hello from thread 1/4 from rank 2/8
Hello from thread 2/4 from rank 2/8
Hello from thread 0/4 from rank 2/8
Hello from thread 3/4 from rank 2/8
Detecting 8 ranks with 4 OMP threads per rank
Hello from thread 0/4 from rank 0/8
Hello from thread 1/4 from rank 0/8
Hello from thread 3/4 from rank 0/8
Hello from thread 2/4 from rank 0/8
Hello from thread 1/4 from rank 1/8
Hello from thread 3/4 from rank 1/8
Hello from thread 0/4 from rank 1/8
Hello from thread 2/4 from rank 1/8
Node r2i6n3 has 4 active GPUs
Node r2i6n3 has 4 active GPUs
Node r2i6n3 has 4 active GPUs
Node r2i6n3 has 4 active GPUs
Hello from Tesla V100-SXM2-16GB GPU 0 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 0 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 0 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 0 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 1 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 1 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 1 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 1 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 2 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 2 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 2 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 2 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 3 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 3 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 3 / 4 on node r2i6n3
Hello from Tesla V100-SXM2-16GB GPU 3 / 4 on node r2i6n3
```
