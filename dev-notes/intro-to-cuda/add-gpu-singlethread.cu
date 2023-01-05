// Following a simple example to port a C++ code to GPU with CUDA.
// https://developer.nvidia.com/blog/even-easier-introduction-cuda/
// Here is the GPU code.

// This kernel is correct for a single thread. Every thread that runs will perform the add on the whole array. This is also a race condition as multiple parallel threads will read and write the same locations.

#include <iostream>
#include <math.h>

// function to add the elements of two arrays

// To port this to GPU, add __global__ to tell the CUDA C++ compiler that this is a function (kernel) to be run on the GPU, called from the CPU code.

// TO run on a GPU, we need to have allocated memory that is accessible by the GPU. Can use "unified memory" to provide a single memory space which is accessible by all GPUs and CPUs in the system.

__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  // Replaced.
  // float *x = new float[N];
  // float *y = new float[N];

   float *x, *y;

   // Allocate Unified Memory -- accessible from CPU or GPU
   cudaMallocManaged(&x, N*sizeof(float));
   cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Replaced
  // // Run kernel on 1M elements on the CPU
  // add(N, x, y);

  // Need to launch the add() kernel to invoke it on the GPU.
  // This launches 1 GPU thread to run add()
  add<<<1, 1>>>(N, x, y);

  // We also need the CPU to wait until the kernel is done before it access the results (CUDA kernel launches do not block CPU thread calls).
  // Wait for GPU to finish before accessing on host
   cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Replaced
  // // Free memory
  // delete [] x;
  // delete [] y;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
