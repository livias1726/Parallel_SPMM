#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>

extern "C" {
#include "../utils/utils.h"
}

#ifndef SCPA_PROJECT_CU_UTILS_H
    #define SCPA_PROJECT_CU_UTILS_H
#endif

#define BD 256
//const dim3 BLOCK_DIM(BD);

void process_arguments(int argc, char **argv, FILE **f, int *k);
int csr_adaptive_blocks(int rows, int *irp, int *blocks);
void allocCudaCsr(CSR *csr, int **d_irp, int **d_ja, double **d_as);
void allocCudaSpmm(double **d_x, double **d_y, const double *x, int m, int n, int k);