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

#define BD 256
#define MAX_REG 65535
#define MAX_THREADS (48 * 1024) // max number of co-resident threads: number of SM * threads per SM
#define MAX_NUM_ROWS 1 // delta number of rows for the Vector kernel computation
#define MAX_K 4
#define MAX_SHARED_MEM = 49152

/**
 *  For a given number of blocks, return a 2D grid large enough to contain them
 *  @param t number of threads (number of rows in scalar kernel)
 *  @param g output grid dimension
 */
#define GET_GRID(t, g)                             \
    if (num_blocks <= MAX_REG){                      \
        g = dim3(num_blocks);                               \
    } else {                                            \
        int side = (int)ceil(sqrt((double)num_blocks)); \
        g = dim3(side,side);                                \
    }

void process_arguments(int argc, char **argv, FILE **f, int *k);
int csr_adaptive_blocks(int rows, int *irp, int *blocks);
void allocCudaCsr(CSR *csr, int **d_irp, int **d_ja, double **d_as);
void allocCudaSpmm(double **d_x, double **d_y, const double *x, int m, int n, int k);