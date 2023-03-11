#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

#ifndef SCPA_PROJECT_CU_UTILS_H
    #define SCPA_PROJECT_CU_UTILS_H
#endif

extern "C"{
#include "../../utils/headers/utils.h"
};

// DEVICE
#define WARP_SIZE 32                        // number of threads in a warp
#define MAX_THREADS_BLOCK 1024              // max number of threads per block
#define SM 48                               // number of available streaming multiprocessors
#define MAX_THREADS SM*MAX_THREADS_BLOCK    // max number of co-resident threads
#define MAX_SHM 48*1024                     // max dimension of the shared memory per block on the specific device

// UTILS
#define FULL_WARP_MASK 0xFFFFFFFF
#define GET_MAX(a, b) if (a < b) {a = b;}
#define GET_SUP_INT(x, y) (x%y != 0) ? x/y + 1 : x/y;

//--------------------------------------------- Signatures ---------------------------------------------------------//
void process_arguments(int argc, char **argv, FILE **f, int *k);
void alloc_cuda_spmm(Type **d_x, Type **d_y, const Type *x, int m, int n, int k);