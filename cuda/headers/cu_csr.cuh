#include "cu_utils.cuh"

#define MAX_NUM_ROWS 1 // delta number of rows for the Vector kernel computation

__global__ void spmm_csr_adaptive_kernel(const int *irp, const int *ja, const Type *as, int k, int max_rows, const Type* x,
                                         int* blocks, Type* y);

void compute_csr_dimensions(CSR* csr, int k, int* blocks, int *num_blocks, dim3* BLOCK_DIM, dim3* GRID_DIM,
                            int *shared_mem, int* max_rows);

void alloc_cuda_csr(CSR *csr, int **d_irp, int **d_ja, Type **d_as);
