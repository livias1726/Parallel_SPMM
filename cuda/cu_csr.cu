#include "headers/cu_csr.cuh"

/**
 * A sub-warp reduction:
 * each thread in signalled in 'mask' participates in the reduction, which starts from offset 's' and terminates on 'k',
 * meaning when each thread in the first sub-warp has accumulated their values from the other sub-warps.
 *
 * In 'spmm_csr_vector_small_even' it's used with the full mask (0xffffffff), meaning each thread in the warp
 * participates. Meanwhile, in 'spmm_csr_vector_small_uneven', the mask is set to include the first 'n' sub-warps,
 * where 'n' is the higher power of two lower or equal than the number of sub-warps.
 *
 * @param mask a 32 bit mask used in __shfl_down_sync to identify the involved threads
 * @param s starting offset
 * @param k sub-warp dimension
 * @param sum value to be reduced
 * */
__device__ Type sub_reduce(unsigned mask, int s, int k, Type sum){
    for(; s >= k; s >>= 1) {
        sum += __shfl_down_sync(mask, sum, s);
    }
    return sum;
}

/*
 * With this configuration there's no need for a reduction phase since each thread does not share the column of x
 * assigned to it. For this reason, each thread's computation is independent, meaning there's no need for explicit
 * synchronization nor usage of shared memory.
 * */
__device__ void spmm_csr_vector_large(const int *irp, const int *ja, const Type *as, int start, int end,
                                      int k, const Type* x, Type* y){

    int kpt = ROUND_UP(k,warpSize);     // number of columns per thread
    int tid_b = threadIdx.x;            // thread id in the block
    int tid_w = tid_b % warpSize;       // thread id in the warp
    int warps = blockDim.x / warpSize;  // number of warps in the block
    int wid = tid_b / warpSize;         // warp id in the block

    int tid_k; // thread's k
    int j, z, r_y;
    Type tmp;

    for (int i = start + wid; i < end; i += warps) { // warp takes the row
        r_y = i * k;

        for (z = 0; z < kpt; z++) {
            tid_k = (z * warpSize) + tid_w;   // column of x each thread will read

            if (tid_k < k) {
                tmp = 0.0;
                for (j = irp[i]; j < irp[i+1]; j++) {
                    // whole warp takes the same non-zero and each thread takes the specific value of x
                    tmp += as[j] * x[ja[j] * k + tid_k];
                }
                y[r_y + tid_k] = tmp;
            }
        }
    }
}

/*
 * Each warp gets divided into warpSize/k sub-warps to perform the products on x's columns in parallel.
 * There's an eventual waste of warpSize % k threads.
 *
 * When warpSize is not multiple of k, there will be a 'truncated' sub-warp, which will be excluded from computation
 * for simplicity. There will also be a number of complete sub-warp that is not a power of two. This will bring a need
 * to perform a pre-reduction to call 'sub_reduce' with a number of sub-warps that is the higher power of two lower than
 * warpSize/k.
 *
 * This implementation needs a shared memory of size BD.
 * */
__device__ void spmm_csr_vector_small(const int *irp, const int *ja, const Type *as, int start, int end,
                                      int k, const Type* x, Type* y){

    extern __shared__ Type LDS[];           // each thread has a cell in the shared memory

    int tid_b = threadIdx.x;                // thread id in the block
    int tid_w = tid_b % warpSize;           // thread id in the warp
    int tid_sw = tid_w % k;                 // thread id in the sub warp
    int wid = tid_b / warpSize;             // warp id in the block
    int swid = tid_w / k;                   // sub warp id in the warp
    int warps = blockDim.x / warpSize;  // number of warps in the block
    int sub_warps = warpSize / k;           // number of sub warps in the warp (truncated one excluded)

    // reduction setup
    int first_pot = 8;
    for (int dig = 3; dig >= 0; dig--) { // find the first power of two lower than the number of sub-warps
        if (sub_warps >> dig) break;
        first_pot >>= 1;
    }
    int s = first_pot * k; // first thread id outside the group of sub-warps used in the reduction
    unsigned mask = __ballot_sync(FULL_WARP_MASK, tid_w < s);
    int excluded = sub_warps - first_pot;

    int j, r_y, sj, ej;
    for (int i = start + wid; i < end; i += warps) { // warp takes the row
        r_y = i * k;
        sj = irp[i];
        ej = irp[i+1];

        LDS[tid_b] = 0.0;

        // ACCUMULATION
        if (swid != sub_warps) { // excludes the truncated sub-warp
            for (j = sj + swid; j < ej; j += sub_warps) { // sub warp takes the non-zero
                // thread in sub warp takes the specific value of x
                LDS[tid_b] += as[j] * x[ja[j] * k + tid_sw];
            }
        }
        //__syncwarp();

        /*
         * REDUCTION 1: values of sub-warps not involved in the warp reduction phase are accumulated
         * in parallel by the other sub-warps.
         * */
        if (swid < excluded) LDS[tid_b] += LDS[tid_b + first_pot * k];
        //__syncwarp();

        // REDUCTION 2
        if (swid < first_pot) LDS[tid_b] = sub_reduce(mask, s>>1, k, LDS[tid_b]);

        // update
        if (swid == 0) y[r_y + tid_sw] = LDS[tid_b];
        //__syncwarp();
    }
}

__device__ void spmm_csr_stream_large(const int *irp, const int *ja, const Type *as, int row_start, int row_end,
                                      int k, const Type* x, Type* y){

    //extern __shared__ Type LDS[]; // each thread has a cell for every nz it should process: tot_nz/sub_blocks

    int tid_b = threadIdx.x;
    int bd = blockDim.x;
    int kpt = ROUND_UP(k, bd);

    Type tmp;
    int z, tid_k;
    int j, sj, ej, r_y;
    for (int i = row_start; i < row_end; i++){ // whole block takes the same row
        r_y = i * k;
        sj = irp[i];
        ej = irp[i + 1];

        for (z = 0; z < kpt; z++) { // each thread computes the product with its columns
            tid_k = (z * bd) + tid_b;

            if (tid_k < k) {
                tmp = 0.0;
                for (j = sj; j < ej; j++) {
                    tmp += as[j] * x[ja[j] * k + tid_k];
                }
                y[r_y + tid_k] = tmp;
            }
        }
    }
}

/*
 * The block gets divided into BD/k sub-blocks to perform the products on x's columns in parallel.
 * There's an eventual waste of BD % k threads.
 *
 * This implementation brings the need of a shared memory of size BD * k, since each thread performs its products until
 * the non-zeros assigned to the block are covered, while the reduction considering the rows is done after.
 * */
__device__ void spmm_csr_stream_small(const int *irp, const int *ja, const Type *as, int row_start, int row_end,
                                      int k, const Type* x, Type* y){

    extern __shared__ Type LDS[]; // each thread has a cell for every nz it should process: tot_nz/sub_blocks

    int first_nz = irp[row_start];
    int tot_nz = irp[row_end] - first_nz;
    int tid_b = threadIdx.x;

    int sub_blocks = blockDim.x / k;
    int sbid = tid_b / k;
    int tid_sb = tid_b % k;
    int lds_row = tid_sb * tot_nz;

    // ACCUMULATION
    if (sbid != sub_blocks) { // needed when bd is not divisible by k
        int idx;
        for (int nz = sbid; nz < tot_nz; nz += sub_blocks) { // each sub-block takes the same nz
            idx = first_nz + nz;
            // each thread in the sub-block computes the product with its column (tid_sb)
            // treats shared memory as a matrix with k rows and tot_nz columns
            LDS[lds_row + nz] = as[idx] * x[ja[idx] * k + tid_sb];
        }
    }
    __syncthreads(); // separates accumulation and reduction for when some threads do not enter the accumulation

    // REDUCTION
    if (sbid != sub_blocks) {
        Type tmp;
        int j;

        for (int i = row_start + sbid; i < row_end; i += sub_blocks){ // each sub-block takes the a row
            tmp = 0.0;
            for (j = (irp[i] - first_nz); j < (irp[i + 1] - first_nz); j++){ // each thread in the sub-block sums up
                tmp += LDS[lds_row + j];
            }
            y[i * k + tid_sb] = tmp;
        }
    }
}

/**
 * VECTOR KERNEL: 1 warp per matrix row.
 *
 * Each warp is divided into warpSize/k sub-blocks to perform the products on x's columns in parallel:
 * - There's an eventual waste of warpSize % k threads.
 * */
__device__ void spmm_csr_vector(const int *irp, const int *ja, const Type *as, int start, int end,
                                int k, const Type* x, Type* y){
    if (k > warpSize >> 1) {
        spmm_csr_vector_large(irp, ja, as, start, end, k, x, y);
    } else {
        spmm_csr_vector_small(irp, ja, as, start, end, k, x, y);
    }
}

/**
 * STREAM KERNEL: streams into the shared memory each individual product related to the NZs assigned to the block
 * */
__device__ void spmm_csr_stream(const int *irp, const int *ja, const Type *as, int row_start, int row_end,
                                int k, const Type* x, Type* y){
    if (k < blockDim.x) {
        spmm_csr_stream_small(irp, ja, as, row_start, row_end, k, x, y);
    } else {
        spmm_csr_stream_large(irp, ja, as, row_start, row_end, k, x, y);
    }
}

/**
 * CSR Adaptive SpMM:
 * dynamically determines whether to execute a set of rows with the stream or the vector kernel.
 *
 * Inspired by Algorithm 3 of Greathouse, Daga - "Efficient Sparse Matrix-Vector Multiplication
 * on GPUs using the CSR Storage Format"
 * */
__global__ void spmm_csr_adaptive_kernel(const int *irp, const int *ja, const Type *as, int k, int max_rows, const Type* x,
                                         int* blocks, Type* y) {

    int block_row_start = blocks[blockIdx.x];
    int block_row_end = blocks[blockIdx.x + 1];
    int rows = block_row_end - block_row_start;

    if (rows > max_rows) { // non-zeros can fit into the LDS
        spmm_csr_stream(irp, ja, as, block_row_start, block_row_end, k, x, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        spmm_csr_vector(irp, ja, as, block_row_start, block_row_end, k, x, y);
    }
}

/**
 * Computes the number of rows to give each block and the total number of blocks to cover all rows.
 *
 * Inspired by Algorithm 2 of Greathouse, Daga - "Efficient Sparse Matrix-Vector Multiplication
 * on GPUs using the CSR Storage Format"
 *
 * NOTE: the loop condition was changed (i <= rows) wrt the original algorithm (i < rows)
 *       to avoid overflowing shared memory when last blocks receives more nnz than it can handle in memory.
 * */
int get_rows_per_block(int bd, int max_rows, int rows, int* irp, int* rows_per_block){

    int nz = 0, last_i = 0, ctr = 1;
    rows_per_block[0] = 0;

    for (int i = 1; i <= rows; i++) {
        nz += irp[i] - irp[i-1]; // count the sum of non-zeros in the considered rows

        // the block can process more non-zeros
        if (nz < bd) continue;

        // there are more non-zeros than threads in a block: decrease number of rows for the block
        if ((nz > bd) && (i - last_i > max_rows)) i-=max_rows;

        // update rows_per_block
        last_i = i;
        rows_per_block[ctr++] = i;
        nz = 0;
    }

    rows_per_block[ctr++] = rows;
    return ctr;
}

void compute_csr_dimensions(CSR* csr, int k, int* blocks, int *num_blocks, dim3* BLOCK_DIM, dim3* GRID_DIM,
                            int *shared_mem, int *max_rows){

    int m = csr->M, nz = csr->NZ, *irp = csr->IRP;

    // 1D block dimension: average number of non-zeros per row
    int bd = ROUND_UP_MULT(ROUND_UP(nz,m), WARP_SIZE); // round up the size of the block to a multiple of warpSize
    if (bd > MAX_THREADS_BLOCK) bd = MAX_THREADS_BLOCK; // check maximum threads per block limit

    // BD defines the limit on the maximum number of rows that discriminates the use of
    // the stream or the vector kernel to avoid wasting some warps using only a fixed number of rows
    int mr = bd / WARP_SIZE;

    // k cannot be higher than 192 when BD is 32
    int mem_k = k * sizeof(Type);
    unsigned int shm = (k < bd) ? bd * mem_k : 0;
    if (shm > MAX_SHM) { // check maximum shared memory per block limit
        shm = MAX_SHM;
        bd = ROUND_DOWN_MULT(shm/mem_k, WARP_SIZE); // reduce block dimension to avoid overflowing shared memory
        mr = bd / WARP_SIZE;
    }

    // 1D grid dimension: compute row balancing on blocks and number of blocks needed
    int nb = get_rows_per_block(bd, mr, m, irp, blocks);

    // output
    *BLOCK_DIM = dim3(bd);
    *max_rows = mr;
    *shared_mem = shm;
    *num_blocks = nb;
    *GRID_DIM = dim3(nb-1);

    printf("Rows = %d, NNZ = %d, BD = %d, GD = %d, SHM = %d\n", m, nz, bd, nb-1, shm);
}