#include "headers/cu_ell.cuh"

__device__ Type sub_reduce(int s, Type sum){
    for(; s > 0; s >>= 1) {
        //if (blockIdx.x == 0 && blockIdx.y == 0) printf("1) %d. T(%d,%d) - %f\n", s, threadIdx.x, threadIdx.y, sum);
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
        //if (blockIdx.x == 0 && blockIdx.y == 0) printf("2) %d. T(%d,%d) - %f\n", s, threadIdx.x, threadIdx.y, sum);
    }
    return sum;
}

/**
* Ellpack SpMM kernel
*
* Each block is given a subset of rows in A (using AS and JA) and a column of x.
* To completely cover A's rows, blocks will eventually need to iterate horizontally on AS and JA and accumulate products.
* */
__global__ void spmm_ell_kernel(int rows, int maxnz, const int *ja, const Type *as, const Type *x, int k, Type* y) {

    //TODO: warp is considered column-wise --> threads with increasing tx and same ty are a part of the same warp
    extern __shared__ Type LDS[];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    const int i = tx + (bdx * bx);  // global row of the thread
    if (i >= rows) return;

    int r_a = i * maxnz;        // ELL's row associated with the thread
    int tid_a = tx * bdy + ty;  // accumulation cell
    int tid_r = ty * bdx + tx;  // reduction cell

    //int wid = tid / warpSize;
    //int warps = (bdx * bdy) / warpSize;

    int idx, j;

    // ACCUMULATION
    /*
     * Each thread takes an element in its row and performs the product with the relative element of x
     * in column given by blockIdx.y
     * */
    LDS[tid_a] = 0.0;
    for (j = ty; j < maxnz; j += bdy) { // do not break the loop when padding reached to avoid mining warp flow
        idx = r_a + j;
        LDS[tid_a] += __dmul_rn(as[idx], x[ja[idx] * k + by]);
    }
    __syncthreads();

    // TODO: try to configure a warp reduction
    // REDUCTION
    // int row_w = warpSize / bdy;
    // since 'bdy' is always a power of 2 <= 32, a warp-level reduction can be executed on the rows
    /*
    LDS[tid_r] = sub_reduce(warpSize>>1, LDS[tid_r]);
    __syncthreads();

    if (ty == 0) y[i * k + by] = LDS[tid_a];
     */

    /*
     * The first thread of each row reduces the partial sums
     * */
    int shift;
    if (ty == 0) {
        for (shift = 1; shift < bdy; shift++) {
            LDS[tid_a] += LDS[tid_a + shift];
        }
        y[i * k + by] = LDS[tid_a];
    }
}

dim3 get_block_dimensions(int m, int maxnz){
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

    return dim3(bx,by);
}

void compute_ell_dimensions(int m, int maxnz, int k, dim3* BLOCK_DIM, dim3* GRID_DIM, int *shared_mem){
    // 2D BLOCK
    dim3 bd = get_block_dimensions(m, maxnz);

    // 2D GRID
    dim3 gd = dim3(ROUND_UP(m,bd.y), k); // number of blocks needed to cover A's rows X number of columns of x

    *shared_mem = bd.x * bd.y * sizeof(Type); // cannot reach maximum shared memory: 1024 * 8 < MAX_SHM
    *BLOCK_DIM = bd;
    *GRID_DIM = gd;

    printf("BLOCK [%d][%d] - GRID [%d][%d] - SHM = %d\n", bd.x, bd.y, gd.x, k, *shared_mem);
}

void alloc_cuda_ell(ELL* ell, int **d_ja, Type **d_as){
    int m = ell->M, maxnz = ell->MAXNZ, dim = m * maxnz;
    int size_ja = dim * sizeof(int), size_as = dim * sizeof(Type);

    int *ja = ell->JA;
    Type *as = ell->AS;

    checkCudaErrors(cudaMalloc((void**) d_ja, size_ja));
    checkCudaErrors(cudaMalloc((void**) d_as, size_as));

    checkCudaErrors(cudaMemcpy(*d_ja, ja, size_ja, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_as, as, size_as, cudaMemcpyHostToDevice));
}