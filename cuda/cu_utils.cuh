#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

extern "C" {
#include "../utils/utils.h"
}

#ifndef SCPA_PROJECT_CU_UTILS_H
    #define SCPA_PROJECT_CU_UTILS_H
#endif

// CUSTOM
#define BDX 256 //TODO use a dynamic dimension wrt the number of NZs
#define MAX_NUM_ROWS 1 // delta number of rows for the Vector kernel computation

// PER DEVICE
#define MAX_THREADS_BLOCK 1024 // max number of threads per block
#define SM_NUM 48 // number of available streaming multiprocessors
#define MAX_THREADS SM_NUM*MAX_THREADS_BLOCK // max number of co-resident threads
#define MAX_SHARED_MEM 49152 // max dimension of the shared memory per block on the specific device

// UTILS
#define FULL_WARP_MASK 0xFFFFFFFF
#define GET_MAX(a, b) if (a < b) {a = b;}

//--------------------------------------------- Signatures ---------------------------------------------------------//
void process_arguments(int argc, char **argv, FILE **f, int *k);
int get_csr_row_blocks(int rows, int *irp, int *blocks, int* max_nz);
int get_shared_memory(int, int);
void alloc_cuda_csr(CSR *csr, int **d_irp, int **d_ja, double **d_as);
void alloc_cuda_spmm(double **d_x, double **d_y, const double *x, int m, int n, int k);