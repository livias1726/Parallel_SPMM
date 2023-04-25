#include "cu_utils.cuh"

/**
 * From 'Three Storage Formats for Sparse Matrices on GPGPUs"
 *
 * Break the original matrix into groups of rows (hacks) and store them as independent matrices in ELLPACK format.
 * The ELLPACK buffers are stacked together inside a 1D array.
 * A 'hackOffsets' vector is used to keep track of the individual sub-matrices.
 *
 * In this implementation, the hacks are groups using blockDim.x, meaning the number of rows given to the block.
 * 'hackOffsets' has size blocks+1. Each element points to the first index of a sub-matrix inside the stacked
 * cM/rP buffers, plus an additional element pointing past the end of the last block.
 * (The elements of the last k-th hack are stored between hackOffsets[k] and hackOffsets[k+1], similarly to CSR.)
 *
 * An additional MAXNZ array is used to keep track of the elements each row in the sub-matrix has.
 * */
typedef struct hll {
    int* HACK_OFFSET;
    int* MAXNZ;         // array of maxnz per block
    int* JA;            // new JA without excessive padding
    Type* AS;           // new AS without excessive padding
} HLL;

__global__ void spmm_hll_kernel(int rows, const int* maxnz, const int* hack_offset,
                                const int *ja, const Type *as, const Type *x, int k, Type* y);
void compute_hll_dimensions(ELL*, int, HLL**, dim3*, dim3*, int*);
unsigned int alloc_cuda_hll(HLL*, int, int**, int**, int**, Type**);
void print_hll(HLL*, int);
