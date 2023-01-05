// -------------------------------
// Hello Hardware - MPI Only
// -------------------------------

// Used to reinstall openmpi with GNU (i.e. working -fopenmp)
// export HOMEBREW_CC=gcc-12
// export HOMEBREW_CXX=g++-12
// brew reinstall openmpi --build-from-source

// Execute (locally on EB-Macbook) using:
// mpicxx -fopenmp hello_ompmpi.cpp
// export OMP_NUM_THREADS=
// mpirun -np 2 ./a.out

#include <iostream>
#include <mpi.h>
#include <omp.h>

int main() {

  MPI_Init(NULL, NULL);  // Initialise MPI
  int rank, world;
  int nthreads, tid;
  int namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // Get my rank
  MPI_Comm_size(MPI_COMM_WORLD, &world);    // Get number of ranks in the world
  MPI_Get_processor_name(processor_name, &namelength);      // Get the name of my processor

  // -------------------------------------------------
  // Print MPI stuff
  // -------------------------------------------------

  if (rank == 0) {
    printf("Total MPI ranks: %d\n", world);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  printf("    Hello from MPI rank: %d from node: %s  \n", rank, processor_name);

  // -------------------------------------------------
  // Print OMP stuff
  // -------------------------------------------------

  // Start a OMP parallel region
  #pragma omp parallel private(tid, nthreads)
  {
    tid = omp_get_thread_num();           // Get my thread ID
    nthreads = omp_get_num_threads();     // Get number of threads in the world

    if (tid == 0 && rank == 0) {
      printf("\nTotal OMP threads per rank: %d\n", nthreads);
    }

    // if (tid == 0)  {
    //   printf("    MPI rank: %d / %d has %d OMP threads\n", rank, world, nthreads);
    // }

    #pragma omp barrier
    printf("    Hello from thread: %d, rank: %d\n", tid, rank);
  }

  MPI_Finalize();

}
