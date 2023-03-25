#include "cu_utils.cuh"

#define BDX 32
#define BDY 32

__global__ void spmm_ell_kernel(int rows, int maxnz, const int *ja, const Type *as, const Type *x, int k, Type* y);
void compute_ell_dimensions(int m, int maxnz, int k, dim3* BLOCK_DIM, dim3* GRID_DIM, int *shared_mem);
void alloc_cuda_ell(ELL* ell, int **d_ja, Type **d_as);

// H-ELLPACK
//HELL* read_mm_hell(Elem** elems, int m, int n, int nz);
//void print_hell(HELL *hell);
