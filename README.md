# Hello Hardware!

Directory containing development of a "hello hardware" code. This prints information about the hardware available to a program given the parameters defined in the submission script. Code available [here](hello-hardware.cu) or see [truncated output](example_short_output.md) and [full output](example_full_output.md) for more information.

<!-- ## To do: -->

**Write a “hello hardware” code to confirm the hardware we are using when launching a submission script.**

<!-- **DONE**

- [x] OpenMP+MPI
- [x] GPU
- [ ] Try to locate cuda code that runs over not only multiple gpu devices but also over multiple gpu nodes.  If it exists, then locate code that cuda uses to determine node count and also how it references unique gpu devices, and then use that code in the hello hardware code; otherwise, simply use #gpus=#gpus_reported_by_cuda x #nodes. Can CUDA find out how many nodes there are?
    Propose to deprecate: no point? -->

***


## Compile:

```
module load nvidia/nvhpc

nvcc -Xcompiler -fopenmp hello-hardware.cu -I/mnt/lustre/indy2lfs/sw/nvidia/hpcsdk-222/Linux_x86_64/22.2/comm_libs/mpi/include -L/mnt/lustre/indy2lfs/sw/nvidia/hpcsdk-222/Linux_x86_64/22.2/comm_libs/mpi/lib -lmpi -lgomp -o hello-gpu

```

```
module load nvidia/nvhpc-nompi
module load mpt

nvcc -Xcompiler -fopenmp hello-hardware.cu -lmpi -lgomp -o hello-gpu-mpt
```



<!--
## Notes:

**How does GPU assignment work on Cirrus?**

The Cirrus GPU nodes contain four Tesla V100-SXM2-16GB cards. Each card has 5,120 CUDA cores and 640 Tensor cores. The ```gpu-cascade``` partition has 36 nodes, each with two 2.5 GHz, 20-core Intel Xeon Gold 6248 (Cascade Lake) series processors.

When using ```--gres=gpu:1``` we are requesting 1 GPU card. By default this will effectively give you 1/4 of the node: 10 CPU cores and ~ 91/384 GB in host memory. You can have access to the entire node by adding the ```--exclusive``` flag. Because these resources are automatically allocated, the submission script does not specify ```ntasks``` or ```cpus-per-task```.

However, jobs are executed on only 1 host process pn 1 core. What if we want to use the remaining cores? These can be allocated via host OpenMP threads. -->
