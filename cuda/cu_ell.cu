#include "headers/cu_ell.cuh"

__device__ double sub_reduce(int s, double sum){
    for(; s > 0; s >>= 1) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
    }
    return sum;
}

/**
* Ellpack SpMM kernel
*
* Each block is given a subset of rows in A (using AS and JA) and a column of x.
* To completely cover A's rows, blocks will eventually need to iterate horizontally on AS and JA and accumulate products.
* */
/*
 * NOTE: see 'get_block_dimensions'.
 */
__global__ void spmm_ell_kernel(int rows, int maxnz, const int *ja, const double *as, const double *x, int k, double* y) {

    extern __shared__ double LDS[];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    const int i = ty + (bdy * bx); //const int i = tx + (bdx * bx);  // global row of the thread
    if (i >= rows) return;

    int r_a = i * maxnz;        // ELL's row associated with the thread
    int tid = tx * bdy + ty;  // accumulation cell

    /* *
     * ACCUMULATION:
     *      Each thread takes an element in its row and performs the product
     *      with the relative element of x in column given by blockIdx.y
     * */
    int idx;
    LDS[tid] = 0.0;
    for (int j = tx; j < maxnz; j += bdx/*int j = ty; j < maxnz; j += bdy*/) { // do not break the loop when padding reached to avoid mining warp flow
        idx = r_a + j;
        LDS[tid] += as[idx] * x[ja[idx] * k + by];
    }
    //__syncthreads();
    __syncwarp();

    /* *
     * REDUCTION:
     *      The first thread of each row reduces the partial sums.
     * */
    LDS[tid] = sub_reduce(bdx>>1, LDS[tid]);
    if (tx == 0) y[i * k + by] = LDS[tid];
    /*
    if (ty == 0) {
        for (idx = 1; idx < bdy; idx++) { LDS[tid] += LDS[tid + idx]; }
        y[i * k + by] = LDS[tid];
    }*/
}

/* *
 * Blocks are dimensioned to cover the Ellpack matrix: x for the rows, y for the columns.
 * Y.
 *      The y dimension is set to be the smaller divisor of warp size that reaches MAXNZ
 *      and set to warp size if MAXNZ is higher than that.
 * X.
 *      The x dimension is set to be a factor of warpSize/blockDim.y,
 *      so that each block will have at least warpSize/blockDim.y rows
 *      to have a block dimension multiple of warpSize.
 *      This factor will be increased to reach the total number of rows or the maximum dimension of the block.
 *
 * NOTE: this configuration causes warp divergence since warps are indexed by threadIdx.x first.
 *       Each thread in the warp will be responsible for a different row, accessing the Ellpack matrix by column.
 * */
dim3 get_block_dimensions(int m, int maxnz){
    // 2D BLOCKS
    int i, max_bx, bx = 0, by;

    // find the smaller number that evenly divides WARP_SIZE and that is higher than maxnz
    for (i = WARP_SIZE >> 1; i > 0; i >>= 1) {
        if (maxnz > i) {
            by = i << 1;
            break;
        }
    }

    // increase by a factor of 'warpSize/blockDim.y' to increase the number of warps in the block
    i = WARP_SIZE / by;
    max_bx = MAX_THREADS_BLOCK / by;
    while (bx < m && bx < max_bx) { bx += i; }

    return dim3(by,bx); //return dim3(bx,by);
}

/**
 * Compute the dimensions of the kernel w.r.t. the number of rows and k.
 *
 * @param m             number of rows
 * @param maxnz         maximum number of non-zeros per row
 * @param k             number of columns in the multi-vector
 * @param block_dim     pointer to the block dimensions
 * @param grid_dim      pointer to the grid dimensions
 * @param shared_mem    pointer to the amount of shared memory
 * */
void compute_ell_dimensions(int m, int maxnz, int k, dim3* block_dim, dim3* grid_dim, int *shared_mem){
    // 2D BLOCK :
    // (#rows given to the block) X (minimum between warpSize and maxnz rounded up to a divisor of warpSize)
    *block_dim = get_block_dimensions(m, maxnz);

    // 2D GRID : (#blocks needed to cover A's rows) X (#columns of x)
    *grid_dim = dim3(ROUND_UP(m, (*block_dim).y), k); //*grid_dim = dim3(ROUND_UP(m, (*block_dim).x), k);

    // 1D SHARED MEMORY :
    // treated like a matrix: 1 cell per block thread
    // cannot reach maximum shared memory thanks to limit on block size (MAX_THREADS_BLOCK * sizeof(double) < MAX_SHM)
    *shared_mem = (*block_dim).x * (*block_dim).y * sizeof(double);
}

/**
 * Allocate and transfer the structures on the device. (ELL version)
 *
 * @param ell           ELL structure
 * @param d_ja          array of column indices (ELL format)
 * @param d_as          array of nz values (ELL format)
 * */
void alloc_cuda_ell(ELL* ell, int **d_ja, double **d_as){
    int dim = ell->M * ell->MAXNZ;
    int size_ja = dim * sizeof(int), size_as = dim * sizeof(double);

    checkCudaErrors(cudaMalloc((void**) d_ja, size_ja));
    checkCudaErrors(cudaMemcpy(*d_ja, ell->JA, size_ja, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) d_as, size_as));
    checkCudaErrors(cudaMemcpy(*d_as, ell->AS, size_as, cudaMemcpyHostToDevice));
}