// -------------------------------
// Hello Hardware - MPI Only
// -------------------------------

// Execute (locally on EB-Macbook) using:
// g++-12 -fopenmp hello.cpp
// export OMP_NUM_THREADS=6
// ./a.out

#include <iostream>
#include <omp.h>

int main() {
  int nthreads, tid;

  // Start a OMP parallel region
  #pragma omp parallel private(tid, nthreads)
  {
    tid = omp_get_thread_num();           // Get my thread ID
    nthreads = omp_get_num_threads();     // Get number of threads in the world

    printf("Hello: thread %d, world: %d\n", tid, nthreads);
  }

  return 0;
}
