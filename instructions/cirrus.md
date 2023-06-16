Cirrus
=========

Compilation instructions for [Cirrus](https://cirrus.readthedocs.io/en/main/), a EPSRC Tier-2 National HPC Facility housed at EPCC's Advanced Computing Facility.

How does GPU assignment work on Cirrus?
-----------------------------------------

The Cirrus GPU nodes contain four Tesla V100-SXM2-16GB cards. Each card has 5,120 CUDA cores and 640 Tensor cores. When using ```--gres=gpu:1``` we are requesting 1 GPU card. By default this will effectively give you 1/4 of the node: 10 CPU cores and ~ 91/384 GB in host memory. You can have access to the entire node by adding the ```--exclusive``` flag. Because these resources are automatically allocated, the submission script does not specify ```ntasks``` or ```cpus-per-task```.


### Obtain the source code:
```
git clone https://github.com/eleanor-broadway/Hello-Hardware.git
cd Hello-Hardware
```

### For NVIDIA MPI:
```
module load nvidia/nvhpc
nvcc -Xcompiler -fopenmp hello-hardware.cu -I$NVHPC_ROOT/comm_libs/mpi/include -L$NVHPC_ROOT/comm_libs/mpi/lib -lmpi -lgomp -o hello-gpu
```

Example submission script:
```
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=01:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=[budget code]

# Load the required module
module load nvidia/nvhpc
cat $0

# Total number of GPUs
NGPUS=1
# Total number of GPUs per node
NGPUS_PER_NODE=1
# Number of CPUs per task
CPUS_PER_TASK=10

export OMP_NUM_THREADS=10
export OMP_PLACES=cores

export SLURM_NTASKS_PER_NODE=${NGPUS_PER_NODE}
export SLURM_TASKS_PER_NODE="${NGPUS_PER_NODE}(x${SLURM_NNODES})"
#export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_mca_base_component_show_load_errors=0

time mpirun -n ${NGPUS} -N ${NGPUS_PER_NODE} ./hello-gpu

```

### For HPE MPT:
```
module load nvidia/nvhpc-nompi
module load mpt

nvcc -Xcompiler -fopenmp hello-hardware.cu -lmpi -lgomp -o hello-gpu-mpt
```

Example submission script:
```
#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=[budget code]

# Load the required modules
module load nvidia/nvhpc-nompi
module load mpt

srun ./hello-gpu-mpt
```
