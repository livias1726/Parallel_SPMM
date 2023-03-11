#include "headers/cu_ell.cuh"

/**
 * Ellpack SpMM kernel
 *
 * Each block is given a subset of rows in A (using AS and JA) and a column of x.
 * To completely cover A's rows, blocks will eventually need to iterate horizontally on AS and JA and accumulate products.
 * */
// blockIdx.x --> gives the row block of A --> blocks with the same Idx.x treats the same rows of A
// blockIdx.y --> gives the column of x --> blocks with the same by treats the same columns of x
// threadIdx.y --> thread's column in AS and JA
__global__ void spmm_ell_kernel(int rows, int maxnz, const int *ja, const Type *as, const Type *x,
                                int k, Type* y) {

    extern __shared__ Type LDS[]; // each thread accumulates its partial result on the row

    const int i = threadIdx.x + (blockDim.x * blockIdx.x); // global row of the thread
    if (i >= rows) return;

    Type nz_val;
    int idx = i * maxnz;
    int sm_base = threadIdx.x * blockDim.x;
    int sm_thread = sm_base + threadIdx.y;
    int j;
    LDS[sm_thread] = 0.0;
    for (int jj = threadIdx.y; jj < maxnz; jj += blockDim.y) {
        idx += jj;
        nz_val = as[idx];
        //if (nz_val == 0) break; --> padding reached but warp flow unbroken
        j = ja[idx];
        LDS[sm_thread] += __dmul_rn(nz_val, x[j * k + blockIdx.y]);
    }
    __syncthreads();

    // reduce
    if (threadIdx.y == 0) { // first thread of each row reduces partial sums
        //if (i == 0 && blockIdx.y == 0) printf("%d\n", maxnz);
        for (int pd=1; pd<blockDim.y; pd++) {
            LDS[sm_base] += LDS[sm_base + pd];
        }
        y[i * k + blockIdx.y] = LDS[sm_base];
    }
}

void alloc_cuda_ell(ELL* ell, int **d_ja, Type **d_as){
    int m = ell->M;
    int maxnz = ell->MAXNZ;
    int size_ja = (m*maxnz)*sizeof(int);
    int size_as = (m*maxnz)*sizeof(Type);

    int *ja = ell->JA;
    Type *as = ell->AS;

    checkCudaErrors(cudaMalloc((void**) d_ja, size_ja));
    checkCudaErrors(cudaMalloc((void**) d_as, size_as));

    checkCudaErrors(cudaMemcpy(*d_ja, ja, size_ja, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_as, as, size_as, cudaMemcpyHostToDevice));
}

void compute_ell_dimensions(int m, int maxnz, int k,
                            dim3* BLOCK_DIM, dim3* GRID_DIM, int *shared_mem){
    // 2D BLOCKS
    *BLOCK_DIM = dim3(BDX,BDY);

    const int gdx = GET_SUP_INT(m,BDX)

    *GRID_DIM = dim3(gdx, k);
    *shared_mem = BDX*BDY*sizeof(Type);
}