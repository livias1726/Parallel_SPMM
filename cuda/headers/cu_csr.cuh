#include "cu_utils.cuh"

__global__ void spmm_csr_adaptive_kernel(const int *irp, const int *ja, const Type *as, int k, const Type* x,
                                         int* blocks, int sm_dim, Type* y);
void compute_csr_dimensions(int m, int k, int *irp, int* blocks, int *num_blocks, dim3* BLOCK_DIM, dim3* GRID_DIM,
                            int *shared_mem);
void alloc_cuda_csr(CSR *csr, int **d_irp, int **d_ja, Type **d_as);
int get_csr_row_blocks(int rows, int *irp, int *blocks, int* max_nz);