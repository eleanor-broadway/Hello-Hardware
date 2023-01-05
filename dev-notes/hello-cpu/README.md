# Development of a MPI+OpenMP "Hello Hardware!" code

Starting from a simple C++ "Hello world!" program and developing individual MPI and OpenMP parallel versions before developing the hybrid OpenMP + MPI code.

Compile and run with:
```
mpicxx -fopenmp hello_ompmpi.cpp -o hello_hybrid

export OMP_NUM_THREADS=2
mpirun -np 2 ./hello_hybird
```
