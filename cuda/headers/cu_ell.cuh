#include "cu_utils.cuh"

// ELLPACK
__global__ void spmm_ell_kernel(int rows, int maxnz, const int *ja, const double *as, const double *x, int k, double* y);
void compute_ell_dimensions(int m, int maxnz, int k, dim3* BLOCK_DIM, dim3* GRID_DIM, int *shared_mem);
void alloc_cuda_ell(ELL* ell, int **d_ja, double **d_as);
