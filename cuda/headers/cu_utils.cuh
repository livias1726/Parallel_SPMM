#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

extern "C"{
#include "../../utils/headers/utils.h"
};

// DEVICE
#define WARP_SIZE 32                        // number of threads in a warp
#define MAX_THREADS_BLOCK 1024              // max number of threads per block
#define MAX_SHM 49152                       // max dimension of the shared memory per block on the specific device (48 kB)
#define MAX_GM 16900423680                  // max global memory (bytes)

// UTILS
#define FULL_WARP_MASK 0xffffffff

//--------------------------------------------- Signatures ---------------------------------------------------------//
void process_arguments(int argc, char **argv, FILE **f, int *k);
unsigned int alloc_cuda_spmm(Type **d_x, Type **d_y, const Type *x, int m, int n, int k);