// Following a simple example to port a C++ code to GPU with CUDA.
// https://developer.nvidia.com/blog/even-easier-introduction-cuda/
// Here is the GPU code.

// How do we make this run in parallel? The key is in CUDA’s <<<1, 1>>>syntax!

#include <iostream>
#include <math.h>

// function to add the elements of two arrays

// STEP 2
// CUDA C++ provides keywords that let the kernels get the indices of running threads.
// threadIdx.x contains the index of the current thread within its block
// blockDim.x contains the number of threads in the block

__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;

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

  // STEP 1
  // This is called the execution configuration and tells the CUDA runtime how many parallel threads to use for the launch on the GPU.
  // The SECOND number in  <<<...>>> is the number of threads in a thread block.
  // CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, 256 is a reasonable size to choose.

  // However, if we only change this line, the code will do the computation once per thread i.e. will not share the addition. The kernel needs to be modified/

  add<<<1, 256>>>(N, x, y);

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
