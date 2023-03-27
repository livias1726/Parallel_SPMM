#include "headers/cu_ell.cuh"

__device__ Type sub_reduce(int s, Type sum){
    for(; s > 0; s >>= 1) {
        //if (blockIdx.x == 0 && blockIdx.y == 0) printf("1) %d. T(%d,%d) - %f\n", s, threadIdx.x, threadIdx.y, sum);
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
        //if (blockIdx.x == 0 && blockIdx.y == 0) printf("2) %d. T(%d,%d) - %f\n", s, threadIdx.x, threadIdx.y, sum);
    }
    return sum;
}

__device__ void spmm_ell(int rows, int start, /*int end,*/ int maxnz, const int *ja, const Type *as, const Type *x, int k, Type* y){

    extern __shared__ Type LDS[];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int bdx = blockDim.x, bdy = blockDim.y;

    const int i = tx + (bdx * bx);  // global row id of the thread
    bool flag = true;
    if (i >= rows) flag = false; // the last block will eventually overflow the total number of rows

    const int s = start + (tx * maxnz); // starting element of the thread
    const int tid = tx * bdy + ty;      // accumulation cell in shared memory

    /* *
     * ACCUMULATION
     *      Each thread takes the elements at (tx, ty (* bdy)), in the hack of the block,
     *      and performs the product with the element of x in column 'by'.
     *      Partial sums gets stored in a matrix like manner: LDS[tx = rows][ty = cols].
     * */
    if (flag) {
        int idx;
        LDS[tid] = 0.0;
        for (int j = ty; j < maxnz; j += bdy) { // do not break the loop when padding reached to avoid mining warp flow
            idx = s + j;
            LDS[tid] += __dmul_rn(as[idx], x[ja[idx] * k + by]);
        }
    }
    __syncthreads();

    /* *
     * REDUCTION
     *      Each row in LDS has to be reduced, but warps are indexed column-wise,
     *      meaning that, if threads take their own cell into the reduction,
     *      values will be reduced by column (sum of values on different rows and same column).
     *
     *      To do a correct reduction, threads need to be responsible for the reversed value: LDS[ty][tx].
     * */
    int tid_r = ty * bdy + tx;
    //if (bx == 504 && by == 0 && ty == 0) printf("T(%d,%d) - tid_a = %d, tid_r = %d\n", tx, ty, tid, tid_r);
    LDS[tid_r] = sub_reduce(warpSize>>1, LDS[tid_r]);
    __syncthreads();
    if (ty == 0) y[i * k + by] = LDS[tid]; // let the first warp take care of the update

    /*
    if (ty == 0) {
        for (idx = 1; idx < bdy; idx++) { LDS[tid] += LDS[tid + idx]; }
        y[i * k + by] = LDS[tid];
    }
     */
}

__global__ void spmm_hll_kernel(int rows, const int* maxnz, const int* hack_offset,
                                const int *ja, const Type *as, const Type *x, int k, Type* y) {

    int mnz = maxnz[blockIdx.x];
    int start = hack_offset[blockIdx.x];
    //int end = hack_offset[bid + 1];

    spmm_ell(rows, start, /*end,*/ mnz, ja, as, x, k, y);
}

dim3 get_block_dimensions(int m, int maxnz){
    // 2D BLOCKS
    int bx, by;

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
    while (bx < m && bx * by < MAX_THREADS_BLOCK) bx += row_w;

    return dim3(bx,by);
}

/*
 * Retrieve the maxnz value for each hack
 * */
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
 * Build the HLL structure from the original ELL and the kernel dimensions.
 *
 * @param ell           original ELL structure
 * @param hll           pointer to the HLL structure to build
 * @param bdx           maximum number of rows per block
 * @param num_blocks    number of blocks to cover every row
 * */
void get_hll(ELL* ell, HLL **hll, int bdx, int num_blocks){
    int m = ell->M, maxnz = ell->MAXNZ;

    int *h_maxnz, *hack_offset, *h_ja;
    Type *h_as;
    // build HLL structure
    *hll = (HLL*) malloc(sizeof(HLL));
    h_maxnz = (int*) malloc(num_blocks * sizeof(int));
    hack_offset = (int*) malloc((num_blocks + 1) * sizeof(int));

    int dim = get_maxnz(m, maxnz, bdx, ell->AS, h_maxnz);   // populate h_maxnz and get new dimension of JA and AS
    h_ja = (int*)calloc(dim, sizeof(int));
    h_as = (Type*)calloc(dim, sizeof(Type));

    int i, j, rs, re, mnz, e_idx, h_idx = 0;
    hack_offset[0] = 0;
    // for every row block re-populate new JA and AS excluding padding overhead
    for (int nb = 0; nb < num_blocks; nb++) {
        mnz = h_maxnz[nb];
        rs = nb * bdx;
        re = MIN(m, rs+bdx);

        for (i = rs; i < re; i++) {
            for (j = 0; j < mnz; j++) {

                e_idx = (i * maxnz) + j;

                h_ja[h_idx] = ell->JA[e_idx];
                h_as[h_idx++] = ell->AS[e_idx];
            }
        }

        hack_offset[nb+1] = hack_offset[nb] + (bdx * mnz);  // populate hack offsets
    }
    hack_offset[num_blocks] = dim;

    // deallocate ELL
    free(ell->JA);
    free(ell->AS);
    free(ell);

    // populate HLL
    (*hll)->MAXNZ = h_maxnz;            // array of maxnz per block
    (*hll)->JA = h_ja;
    (*hll)->AS = h_as;
    (*hll)->HACK_OFFSET = hack_offset;
}

/**
 * Compute the dimensions of the kernel w.r.t. the number of rows and k and builds the HLL structure starting from
 * these dimensions and the original ELL structure.
 *
 * @param ell           original ELL structure
 * @param k             number of columns in the multi-vector
 * @param hll           pointer to the HLL structure to build
 * @param BLOCK_DIM     pointer to the block dimensions
 * @param GRID_DIM      pointer to the grid dimensions
 * @param shared_mem    pointer to the amount of shared memory
 * */
void compute_hll_dimensions(ELL* ell, int k, HLL **hll, dim3* BLOCK_DIM, dim3* GRID_DIM, int *shared_mem){

    int m = ell->M, maxnz = ell->MAXNZ;
    // 2D BLOCK :
    // (#rows given to the block) X (minimum between warpSize and maxnz rounded up to a divisor of warpSize)
    dim3 bd = get_block_dimensions(m, maxnz);
    // 2D GRID : (#blocks needed to cover A's rows) X (#columns of x)
    dim3 gd = dim3(ROUND_UP(m,bd.x), k);

    // build the HLL structure
    get_hll(ell, hll, bd.x, gd.x);

    // 1D SHARED MEM treated like a matrix: 1 cell per block thread
    // cannot reach maximum shared memory thanks to limit on block size (MAX_THREADS_BLOCK * sizeof(Type) < MAX_SHM)
    *shared_mem = bd.x * bd.y * sizeof(Type);

    *BLOCK_DIM = bd;
    *GRID_DIM = gd;
}

/**
 * Allocate and transfer the structures on the device. (HLL version)
 *
 * @param hll           HLL structure previously built
 * @param num_blocks    the number of blocks that takes the different rows (x side of the grid)
 * @param d_maxnz       the array of maxnz per hack
 * @param d_hack        the array of hack offsets
 * @param d_ja          the array of column indices (ELL format)
 * @param d_as          the array of nz values (ELL format)
 * */
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