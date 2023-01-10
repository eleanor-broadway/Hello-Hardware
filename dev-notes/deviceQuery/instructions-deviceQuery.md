# Using the deviceQuery code:

```
PRFX=/work/z04/z04/ebroadwa/CompBioMed/CUDA-test/
cd $PRFX

module load nvidia/nvhpc

git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery
```
Need to update line 35 in the Makefile to ```CUDA_PATH ?= ${NVHPC_ROOT}/cuda```

```
make TARGET_ARCH=x86_64
```
