#include "headers/cu_csr.cuh"

/**
 * A sub-warp reduction: each thread in signalled in 'mask' participates in the reduction, which starts from offset 's'.
 * The reduction terminates using 'k', meaning when each thread in the first sub-warp has accumulated the relative values
 * from the other sub-warps.
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

/**
  * VECTOR KERNEL with k >= warpSize: 1 warp per matrix row
  * (with bd=MAX, needed a k x7 bigger (> 192) than warpSize to reach over MAX_SHM)
  * */
__device__ void spmm_csr_vector_large(const int *irp, const int *ja, const Type *as, int start, int end,
                                      int k, const Type* x, Type* y){

    /*
     * With this configuration there's no need for a reduction phase since each thread takes care of a value of y
     * given by the row taken by the warp and the column given by the k assigned iteratively.
     *
     * Since each thread's computation is not dependent from the others,
     * there's also no need for explicit synchronization.
     * */

    // each thread in the warp uses ceil(k/warpSize) cells in LDS to store accumulation
    extern __shared__ Type LDS[];
    int lds_per_thread = ROUND_UP(k,warpSize);

    int tid_b = threadIdx.x;            // thread id in the block
    int tid_w = tid_b % warpSize;       // thread id in the warp
    int warps = blockDim.x / warpSize;  // number of warps in block
    int wid = tid_b / warpSize;         // warp id

    int k_lds, tid_k; // thread's k
    int j, lds, r_y, sj, ej;
    Type val_a, val_x;

    for (int i = start + wid; i < end; i += warps) { // warp takes the row
        r_y = i * k;
        sj = irp[i];
        ej = irp[i+1];

        for (lds = 0; lds < lds_per_thread; lds++) {
            k_lds = (lds * blockDim.x) + tid_b; // cell of LDS each thread will update
            tid_k = (lds * warpSize) + tid_w;   // column of x each thread will read

            LDS[k_lds] = 0.0;

            if (tid_k < k) {
                for (j = sj; j < ej; j++) {
                    // whole warp takes the same non-zero and each thread takes the specific value of x
                    val_a = as[j];
                    val_x = x[ja[j] * k + tid_k];

                    LDS[k_lds] += val_a * val_x;
                }

                y[r_y + tid_k] = LDS[k_lds];
            }
        }
    }
}

/**
 * VECTOR KERNEL with k < warpSize: 1 warp per matrix row
 * */
__device__ void spmm_csr_vector_small(const int *irp, const int *ja, const Type *as, int start, int end,
                                      int k, const Type* x, Type* y){

    extern __shared__ Type LDS[];           // each thread has a cell in the shared memory

    int tid_b = threadIdx.x;                // thread id in the block
    int tid_w = tid_b % warpSize;           // thread id in the warp
    int tid_sw = tid_w % k;                 // thread id in the sub warp
    int wid = tid_b / warpSize;             // warp id
    int swid = tid_w / k;                   // sub warp id
    int num_warps = blockDim.x / warpSize;  // number of warps in the block
    int sub_warps = warpSize / k;   // number of sub warps in the warp (truncated excluded)

    // reduction setup
    int first_pot = 8;
    for (int dig = 3; dig >= 0; dig--) {
        if (sub_warps >> dig) break;
        first_pot >>= 1;
    }
    int s = first_pot * k; // first thread id outside the group of sub-warps used in the reduction
    unsigned mask = __ballot_sync(FULL_WARP_MASK, tid_w < s);
    int excluded = sub_warps - first_pot;

    int j, r_y, sj, ej;
    for (int i = start + wid; i < end; i += num_warps) { // warp takes the row
        r_y = i * k;
        sj = irp[i];
        ej = irp[i+1];

        LDS[tid_b] = 0.0;

        // ACCUMULATION
        if (swid != sub_warps) {
            for (j = sj + swid; j < ej; j += sub_warps) { // sub warp takes the non-zero
                // thread in sub warp takes the specific value of x
                LDS[tid_b] += as[j] * x[ja[j] * k + tid_sw];
            }
        }
        //__syncwarp();

        // REDUCTION PHASE 1:
        // values of sub-warps not involved in the warp reduction phase
        // are accumulated in parallel by the other sub-warps.
        if (swid < excluded) LDS[tid_b] += LDS[tid_b + first_pot * k];
        //__syncwarp();

        // REDUCTION PHASE 2
        if (swid < first_pot) LDS[tid_b] = sub_reduce(mask, s>>1, k, LDS[tid_b]);

        // update
        if (swid == 0) y[r_y + tid_sw] = LDS[tid_b];
        //__syncwarp();
    }
}

/**
 * STREAM KERNEL: streams into the shared memory each individual product related to the NZs assigned to the block.
 *
 * The block gets divided into BD/k sub-blocks to perform the products on x's columns in parallel:
 * - There's a waste of BD % k threads.
 *
 * TODO: manage the case when k > BD --> BD/k = 0
 * */
__device__ void spmm_csr_stream(const int *irp, const int *ja, const Type *as, int row_start, int row_end,
                                       int k, const Type* x, Type* y){

    extern __shared__ Type LDS[]; // each thread has a cell for every nz it should process: tot_nz/sub_blocks

    int first_nz = irp[row_start];
    int tot_nz = irp[row_end] - first_nz;
    int tid_b = threadIdx.x;

    int sub_blocks = blockDim.x / k;
    int sbid = tid_b / k;
    int tid_sb = tid_b % k;

    // ACCUMULATION 1
    if (sbid != sub_blocks) { // needed when bd is not divisible by k
        Type val_a, val_x;
        int nz, idx;
        for (nz = sbid; nz < tot_nz; nz += sub_blocks) { // each sub-block takes the same nz
            idx = first_nz+nz;
            val_a = as[idx];
            val_x = x[ja[idx] * k + tid_sb]; // each thread in the sub-block computes the product with its column (tid_sb)

            // treats shared memory as a matrix with k rows and tot_nz columns
            LDS[tid_sb * tot_nz + nz] = val_a * val_x;
        }
    }
    __syncthreads(); // separates accumulation and reduction for when some threads do not enter the accumulation

    // REDUCTION
    if (sbid != sub_blocks) {
        Type tmp;
        int j, start, end, r_y;
        for (int i = row_start + sbid; i < row_end; i += sub_blocks){ // each sub-block takes the a row
            r_y = i * k;

            start = irp[i] - first_nz;
            end = irp[i + 1] - first_nz;
            tmp = 0.0;

            for (j = start; j < end; j++){ // each thread in the sub-block sums up
                tmp += LDS[tid_sb * tot_nz + j];
            }

            y[r_y + tid_sb] = tmp;
        }
    }
    //__syncthreads();
}

/**
 * CSR Adaptive SpMM: dynamically determines whether to execute a set of rows with the stream or the vector kernel.
 *
 * Inspired by Algorithm 3 of Greathouse and Daga's
 * "Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format"
 * */
__global__ void spmm_csr_adaptive_kernel(const int *irp, const int *ja, const Type *as, int k, const Type* x,
                                         int* blocks, Type* y) {


    int block_row_start = blocks[blockIdx.x];
    int block_row_end = blocks[blockIdx.x + 1];
    int rows = block_row_end - block_row_start;

    if (rows > MAX_NUM_ROWS) { // non-zeros can fit into the LDS
        if (k > blockDim.x) { //TODO
            printf("CANNOT YET PERFORM THIS SPMM\n");
            return;
        }

        spmm_csr_stream(irp, ja, as, block_row_start, block_row_end, k, x, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        if (k >= warpSize >> 1) {
            spmm_csr_vector_large(irp, ja, as, block_row_start, block_row_end, k, x, y);
        } else {
            spmm_csr_vector_small(irp, ja, as, block_row_start, block_row_end, k, x, y);
        }
    }
}

/**
 * Computes the number of rows to give each block (s.t. # NZ <= BD) and the total number of blocks to cover all rows.
 *
 * Inspired by Algorithm 2 of Greathouse and Daga's
 * "Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format".
 *
 * @param rows total number of rows in A
 * @param irp row delimiters of CSR format
 * @param blocks output array of row blocks
 * */
int get_csr_row_blocks(int bd, int rows, int* irp, int* rows_per_block){

    int nz = 0, last_i = 0, ctr = 1;
    rows_per_block[0] = 0;

    for (int i = 1; i < rows; i++) {
        nz += irp[i] - irp[i-1]; // count the sum of non-zeros in the considered rows

        if (nz < bd) continue; // the block can process more non-zeros

        if ((nz > bd) && (i - last_i > 1)) { // not enough space
            // there are more non-zeros than threads in a block AND
            // more than 1 row was scanned for the block: decrease number of rows for the block
            i--;
        }

        last_i = i;
        rows_per_block[ctr++] = i;
        nz = 0;
    }

    rows_per_block[ctr++] = rows;
    return ctr;
}

void compute_csr_dimensions(int m, int nz, int k, int *irp, int* blocks, int *num_blocks, dim3* BLOCK_DIM, dim3* GRID_DIM,
                            int *shared_mem){
    // 1D block dimension
    // Average number of non-zeros per row: this may not be the best dimension in every case,
    // but it is statistically appropriate to manage the shared memory.
    int avg = ROUND_UP(nz,m);
    int bd = ROUND_UP_MULT(avg,WARP_SIZE); // round up the size of the block to a multiple of warp size
    if (bd > MAX_THREADS_BLOCK) bd = MAX_THREADS_BLOCK; // check maximum threads per block limit
    *BLOCK_DIM = dim3(bd);

    // 1D grid dimension: compute row balancing on blocks and number of blocks needed
    // TODO: consider k in balancing
    *num_blocks = get_csr_row_blocks(bd, m, irp, blocks);
    *GRID_DIM = dim3(*num_blocks-1);

    // TODO: manage max shm
    // compute shared memory dimension
    // if vector kernel:
    int k_factor = ROUND_UP(k, WARP_SIZE); // number of columns assigned to each thread
    *shared_mem = bd * k_factor * sizeof(Type);
    // if stream kernel:

    //*shared_mem = bd * k * sizeof(Type); // with this: max k < MAX_SHM/(bd * sizeof(Type))
}