// -------------------------------
// Hello Hardware - MPI Only
// -------------------------------

// Execute using:
// mpicxx hello_mpi.cpp
// mpirun -np 2 ./a.out

#include <iostream>
#include <mpi.h>

int main() {

  MPI_Init(NULL, NULL);  // Initialise MPI
  int rank, world;
  int namelength;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // Get my rank
  MPI_Comm_size(MPI_COMM_WORLD, &world);    // Get number of ranks in the world
  MPI_Get_processor_name(processor_name, &namelength);      // Get the name of my processor

  printf("Hello: rank %d, world: %d, processor: %s\n",rank, world, processor_name);

  MPI_Finalize();

}
