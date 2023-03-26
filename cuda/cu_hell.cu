#include "headers/cu_ell.cuh"

__device__ Type sub_reduce(int s, Type sum){
    for(; s > 0; s >>= 1) {
        //if (blockIdx.x == 0 && blockIdx.y == 0) printf("1) %d. T(%d,%d) - %f\n", s, threadIdx.x, threadIdx.y, sum);
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
        //if (blockIdx.x == 0 && blockIdx.y == 0) printf("2) %d. T(%d,%d) - %f\n", s, threadIdx.x, threadIdx.y, sum);
    }
    return sum;
}

__device__ void spmm_ell(int start, int end, int maxnz, const int *ja, const Type *as, const Type *x, int k, Type* y){

    extern __shared__ Type LDS[];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    const int i = tx + (bdx * bx);  // global row of the thread
    if (i * maxnz >= end) return;

    int tid_a = tx * bdy + ty;  // accumulation cell

    // ACCUMULATION
    /*
    * Each thread takes an element in its row and performs the product with the relative element of x
    * in column given by blockIdx.y
    * */
    int idx, j;
    LDS[tid_a] = 0.0;
    for (j = ty; j < maxnz; j += bdy) { // do not break the loop when padding reached to avoid mining warp flow
        idx = i + j;
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

__global__ void spmm_hll_kernel(int rows, const int* maxnz, const int* hack_offset,
                                const int *ja, const Type *as, const Type *x, int k, Type* y) {
    int bid = blockIdx.x;
    int mnz = maxnz[bid];
    int start = hack_offset[bid];
    int end = hack_offset[bid + 1];

    spmm_ell(start, end, mnz, ja, as, x, k, y);
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

int get_maxnz(int rows, int cols, int rb, Type* as, int *mnz){
    int i, j, new_dim, row;
    int b_ctr = 0, nz_ctr = 0, max = 0;

    int s = 0, e;
    do {
        e = MIN(rows,s+rb);
        for (i = s; i < e; i++) {
            row = i * cols;
            for (j = 0; j < cols; j++) {
                if (as[row + j] == 0) break;
                nz_ctr++;
            }
            if (nz_ctr > max) max = nz_ctr;
            nz_ctr = 0;
        }

        mnz[b_ctr] = max;       // save maxnz for block
        b_ctr++;                // increase block counter
        new_dim += max * (e-s); // compute new arrays dimension
        max = 0;
        s += rb;
    } while (s < rows);

    return new_dim;
}

/**
 * TODO
 * */
void get_hll(ELL* ell, HLL **hll, int bdx, int num_blocks){
    int m = ell->M, maxnz = ell->MAXNZ;
    int *h_maxnz, *hack_offset, *h_ja;
    Type *h_as;

    // build HLL structure
    *hll = (HLL*) malloc(sizeof(HLL));

    h_maxnz = (int*) malloc(num_blocks * sizeof(int));
    int dim = get_maxnz(m, maxnz, bdx, ell->AS, h_maxnz);

    h_ja = (int*)calloc(dim, sizeof(int));
    h_as = (Type*)calloc(dim, sizeof(Type));
    hack_offset = (int*) malloc((num_blocks + 1) * sizeof(int));

    int i, j, rs, re, mnz, idx1, idx2;
    for (int nb = 0; nb < num_blocks; nb++) {
        mnz = h_maxnz[nb];
        rs = nb * bdx;
        re = MIN(m, rs+bdx);

        for (i = rs; i < re; i++) {
            for (j = 0; j < mnz; j++) {
                idx1 = (i * maxnz) + j;
                idx2 = (i * mnz) + j;

                h_ja[idx2] = ell->JA[idx1];
                h_as[idx2] = ell->AS[idx1];
            }
        }
        hack_offset[nb] = (nb == 0) ? 0 : rs * h_maxnz[nb-1];
    }
    hack_offset[num_blocks] = dim;

    // deallocate ELL
    free(ell->JA);
    free(ell->AS);
    free(ell);

    // populate HLL
    (*hll)->MAXNZ = h_maxnz;
    (*hll)->JA = h_ja;
    (*hll)->AS = h_as;
    (*hll)->HACK_OFFSET = hack_offset;
}

void compute_hll_dimensions(ELL* ell, int k, HLL **hll, dim3* BLOCK_DIM, dim3* GRID_DIM, int *shared_mem){
    int m = ell->M, maxnz = ell->MAXNZ;

    // 2D BLOCK
    dim3 bd = get_block_dimensions(m, maxnz);
    // 2D GRID
    dim3 gd = dim3(ROUND_UP(m,bd.x), k); // number of blocks needed to cover A's rows X number of columns of x

    // build the HLL structure
    get_hll(ell, hll, bd.x, gd.x);

    // 1 cell of shared memory per thread
    *shared_mem = bd.x * bd.y * sizeof(Type); // cannot reach maximum shared memory: 1024 * 8 < MAX_SHM
    *BLOCK_DIM = bd;
    *GRID_DIM = gd;
}

void alloc_cuda_hll(HLL* hll, int num_blocks, int **d_maxnz, int **d_hack, int **d_ja, Type **d_as){
    int *maxnz = hll->MAXNZ, *ja = hll->JA, *hack = hll->HACK_OFFSET;
    Type *as = hll->AS;

    int size_mnz = num_blocks * sizeof(int);
    int size_ho = (num_blocks + 1) * sizeof(int);
    int size_ja = hack[num_blocks] * sizeof(int);
    int size_as = hack[num_blocks] * sizeof(Type);

    checkCudaErrors(cudaMalloc((void**) d_maxnz, size_mnz));
    checkCudaErrors(cudaMalloc((void**) d_hack, size_ho));
    checkCudaErrors(cudaMalloc((void**) d_ja, size_ja));
    checkCudaErrors(cudaMalloc((void**) d_as, size_as));

    checkCudaErrors(cudaMemcpy(*d_maxnz, maxnz, size_mnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_hack, hack, size_ho, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(*d_ja, ja, size_ja, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_as, as, size_as, cudaMemcpyHostToDevice));
}

void print_hll(HLL* hll){
    /*TODO*/
}