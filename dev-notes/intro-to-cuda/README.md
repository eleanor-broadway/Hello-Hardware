# Intro to CUDA

Following a simple example to [port a C++ code to GPU with CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/).

This directory takes the user from a serial C++ program, ```add.cpp```, to a GPU parallel code. Moving from single thread, ```add-gpu-singlethread.cu```, to threaded parallelism, ```add-gpu-threads.cu```, to defining a grid of threads and blocks, ```add-gpu-grid.cu```.

## Notes on CUDA programming:

[Getting Started with CUDA](https://developer.nvidia.com/blog/cuda-refresher-getting-started-with-cuda/)

[CUDA programming model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)

The programmer:
- Divides problems into small independent problems which can be solved independently in **CUDA blocks**.
- CUDA blocks are a collection or group of threads.
- Each block solves the sub-problem with parallel threads executing and cooperating with each other.
- The number of threads/blocks are defined by the programmer and each have a unique global id, also assigned by the programmer.
- Blocks are grouped into grids.
- Therefore a kernel is executed as a grid of blocks of threads.

This program is submitted to the GPU:
- A GPU has *X* SMs (streaming multiprocessors).
- The CUDA **runtime** schedules the blocks on multiprocessors in a GPU in any order.
- A thread is executed on a core.
- A thread block is executed on a SM.
- 1 SM can run several concurrent blocks.
- A kernel grid is executed on a GPU.
- Therefore the program can scale and run on any number of multiprocessors. (For a smaller GPU with four SMs, each SM gets two CUDA blocks. For a larger GPU with eight SMs, each SM gets one CUDA block).

The host code is running a C++ program on the CPU and the kernel (device code) is running on a physically separate GPU device.
- Each kernel is executed on one device and CUDA supports running multiple kernels on a device at one time
- CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
- The dimension of the thread block is accessible within the kernel through the built-in blockDim variable.
- The version number of a GPU can be used by applications to determine hardware features or instructions available on the present GPU.
- The ```deviceQuery``` CUDA sample code can be used to see the properties of CUDA devices present in the system.

Executing a CUDA program:
- Copy input data from host memory to device memory
- Load GPU program and execute
- Copy results from device memory to host memory

Q: How many SMs do the Cirrus GPUs have?
Q: How do these principals differ when using C++ to parallelise for GPU?
