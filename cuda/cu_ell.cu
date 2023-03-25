#include "headers/cu_ell.cuh"

__device__ Type sub_reduce(int s, int end, Type sum){
    for(; s >= end; s >>= 1) {
        if (blockIdx.x == 0 && blockIdx.y == 0) printf("1) %d. T(%d,%d) - %f\n", s, threadIdx.x, threadIdx.y, sum);
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
        if (blockIdx.x == 0 && blockIdx.y == 0) printf("2) %d. T(%d,%d) - %f\n", s, threadIdx.x, threadIdx.y, sum);
    }
    return sum;
}

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

    extern __shared__ Type LDS[];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    const int i = tx + (bdx * bx);  // global row of the thread
    if (i >= rows) return;

    int r_a = i * maxnz;            // ELL's row associated with the thread
    int tid_shm = tx * bdy + ty;    // LDS cell related to each thread

    // int wid = tid_shm / warpSize;
    // int warps = (bdx * bdy) / warpSize;

    Type val_a, val_x;
    int idx, j;

    // ACCUMULATION
    LDS[tid_shm] = 0.0;
    for (j = ty; j < maxnz; j += bdy) {
        idx = r_a + j;
        val_a = as[idx];
        val_x = x[ja[idx] * k + by];
        //if (nz_val == 0) break; --> padding reached but warp flow unbroken
        LDS[tid_shm] += __dmul_rn(val_a, val_x);
    }
    __syncthreads();

    // TODO: try to configure a warp reduction
    // REDUCTION
    /*
    int row_w = warpSize / bdy;
    // since 'bdy' is always a power of 2 <= 32, a warp-level reduction can be executed on the rows
    LDS[tid_shm] = sub_reduce(warpSize>>1, row_w, LDS[tid_shm]);
    if (ty == 0) y[i * k + by] = LDS[tid_shm];
     */

    if (ty == 0) { // first thread of each row reduces partial sums
        for (int pd = 1; pd < bdy; pd++) {
            LDS[tid_shm] += LDS[tid_shm + pd];
        }
        y[i * k + by] = LDS[tid_shm];
    }
}

void compute_ell_dimensions(int m, int maxnz, int k, dim3* BLOCK_DIM, dim3* GRID_DIM, int *shared_mem){
    // 2D BLOCKS
    int bx, by; // number of rows and threads per
    // by -> find the smaller number that evenly divides WARP_SIZE that is higher than maxnz
    for (int i = WARP_SIZE >> 1; i > 0; i >>= 1) {
        if (maxnz > i) {
            by = i << 1;
            break;
        }
    }
    // bx -> each block will have at least WARP_SIZE / by rows to have BD multiple of WARP_SIZE
    int row_w = WARP_SIZE / by;
    bx = row_w;
    // increase by a factor of 'row_w' to increase the number of warps in the block
    while (bx < m && bx * by < MAX_THREADS_BLOCK) {
        bx += row_w;
    }

    *BLOCK_DIM = dim3(bx,by);

    int gdx = ROUND_UP(m,by); // number of blocks needed to cover A's rows
    *GRID_DIM = dim3(gdx,k);

    // Cannot reach maximum shared memory: 1024 * 8 < MAX_SHM
    *shared_mem = bx * by * sizeof(Type);

    printf("BLOCK [%d][%d] - GRID [%d][%d] - SHM = %d\n", bx, by, gdx, k, *shared_mem);
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