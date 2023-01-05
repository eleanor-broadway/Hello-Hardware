// Following a simple example to port a C++ code to GPU with CUDA.
// https://developer.nvidia.com/blog/even-easier-introduction-cuda/
// Here is the GPU code.

// CUDA GPUs have parallel processors grouped into Streaming Multiprocessors, SMs.
// Each SM can run multiple concurrent thread blocks.
// Example: Tesla P100 GOU has 56 SMs, each able to support up to 2048 active threads. To take advantage of this, can launch the kernrl with multiple thread blocks.

#include <iostream>
#include <math.h>

// function to add the elements of two arrays

// CUDA C++ provides keywords that let the kernels get the indices of running threads.
// threadIdx.x contains the index of the current thread within its block
// blockDim.x contains the number of threads in the block
// gridDim.x contains the number of blocks in the grid
// blockIdx.x contains the index or the current thread block in the grid

// Each thread gets an index by computing the offset to the beginning of its block (the block index times the block size: blockIdx.x * blockDim.x) and adding the thread’s index within the block (threadIdx.x).

__global__
void add(int n, float *x, float *y)
{
  // int index = threadIdx.x;
  // int stride = blockDim.x;

  int index = blockIdx.x * blockDim.x + threadIdx.x;    // Thread index
  int stride = blockDim.x * gridDim.x;    // Set to the total number of threads in the grid

  // for (int i = 0; i < n; i++)
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

   float *x, *y;

   // Allocate Unified Memory -- accessible from CPU or GPU
   cudaMallocManaged(&x, N*sizeof(float));
   cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Need to launch the add() kernel to invoke it on the GPU.

  // This is called the execution configuration and tells the CUDA runtime how many parallel threads to use for the launch on the GPU.
  // The FIRST number in <<<...>>> is the number of thread blocks.
  // The SECOND number is the number of threads in a thread block.

  // These blocks of parallel threads make up a grid.

  // Have N elements to process and 256 threads per block, need to calculate the number of blocks to get at least N threads.

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
   cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
