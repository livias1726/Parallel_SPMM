#include "cu_utils.cuh"

__global__ void spmm_csr_vector_kernel(const int *irp, const int *ja, const double *as, int k, const double* x,
                                         int* blocks, double* y);

void compute_csr_dimensions(CSR* csr, int k, int* blocks, int *num_blocks, dim3* BLOCK_DIM, dim3* GRID_DIM,
                            int *shared_mem);

void alloc_cuda_csr(CSR*, int*, int, int**, int**, double**, int**);
