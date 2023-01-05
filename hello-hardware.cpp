// -------------------------------
// Hello Hardware
// -------------------------------

#include <iostream>
#include <mpi.h>
#include <omp.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

int main(void) {

  MPI_Init(NULL, NULL);  // Initialise MPI
  int rank, world;
  int nthreads, tid;
  int namelength;
  int nDevices;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // Get my rank
  MPI_Comm_size(MPI_COMM_WORLD, &world);    // Get number of ranks in the world
  MPI_Get_processor_name(processor_name, &namelength);      // Get the name of my processor

  // Current rank queries it's available hardware for the number of GPU devices
  // i.e. the number of GPU devices on this ranks' processor.
  // cudaGetDeviceCount(&nDevices);

  //---------------------------------------------
  // Uncomment to print hello from each MPI rank
  //---------------------------------------------
  // printf("Hello from rank %d/%d\n", rank, world);

  #pragma omp parallel private(tid, nthreads)
  {
    tid = omp_get_thread_num();           // Get my thread ID
    nthreads = omp_get_num_threads();     // Get number of threads in the world

    // Print global stuff
    if (tid == 0 && rank == 0){
      printf("Detecting %d ranks with %d OMP threads per rank\n", world, nthreads);
    }

    //-----------------------------------------------
    // Uncomment to print hello from each OMP thread
    //-----------------------------------------------
    // for (int i = 0; i < nthreads; i++){
    //   if (tid == i){
    //     printf("Hello from thread %d/%d from rank %d/%d\n", tid, nthreads, rank, world);
    //   }
    // }

  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Print local to node stuff

  // for (int i = 0; i < world; i++){
  //   if (rank == i){
  //     if (processor_name == previous_processor_name){
  //       exit loop
  //     }
  //     and if processor_name .ne. previous_processor_names
  //     printf("Node %s has %d active GPUs\n", processor_name, nDevices);
  //     processor_name = used_processor_name
  //   }
  // }
  //
  //
  // OR
  //
  // have a vector with all of the strings
  // Sort



  // Testing...
  // send_data, send_count, MPI_Datatype, recv_data, recv_count, MPI_Datatype, root, MPI_COMM_WORLD
  MPI_Gather (processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, total[rank+1], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

  if (rank==0){

    printf(total)

  }






  //------------------------------------------------------
  // Uncomment to print information about each GPU thread
  //------------------------------------------------------
  // for (int i = 0; i < nDevices; i++) {
  //   cudaDeviceProp prop;
  //   cudaGetDeviceProperties(&prop, i);
  //   printf("Hello from %s GPU %d / %d on node %s \n", prop.name, i, nDevices, processor_name);
  // }

  MPI_Finalize();
  return 0;

}
