#include "headers/cu_csr.cuh"

__device__ Type warp_reduce(Type sum){
    // implementation of a logarithmic reduction with warp-level communication primitive
    for(int s = warpSize >> 1; s > 0; s >>= 1) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
    }
    return sum;
}

__device__ Type sub_reduce(unsigned mask, int s, int k, Type sum){
    for(; s >= k; s >>= 1) {
        sum += __shfl_down_sync(mask, sum, s);
    }
    return sum;
}

/**
  * VECTOR KERNEL with k >= warpSize: 1 warp per matrix row.
  * (with bd=MAX, needed a k x7 bigger (> 192) than warpSize to reach over MAX_SHM)
  * */
__device__ void spmm_csr_vector_large(const int *irp, const int *ja, const Type *as, int start, int end,
                                int k, const Type* x, Type* y){
    // use of shared memory
    extern __shared__ Type LDS[];

    // each thread in the warp uses ceil(k/warpSize) cells in LDS to store accumulation
    int lds_per_thread = ROUND_UP(k,warpSize);

    int tid_b = threadIdx.x;            // thread id in the block
    int tid_w = tid_b % warpSize;       // thread id in the warp
    int warps = blockDim.x / warpSize;  // number of warps in block
    int wid = tid_b / warpSize;         // warp id in the block

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
        __syncwarp();
    }
}

/**
  * VECTOR KERNEL with k < warpSize and warpSize multiple of k: 1 warp per matrix row.
  * */
__device__ void spmm_csr_vector_small_even(const int *irp, const int *ja, const Type *as, int start, int end,
                                int k, const Type* x, Type* y){

    extern __shared__ Type LDS[];

    int tid_b = threadIdx.x, tid_w = tid_b % warpSize, tid_sw = tid_b % k;
    int num_warps = blockDim.x / warpSize, sub_warps = ROUND_UP(warpSize,k);
    int wid = tid_b / warpSize, swid = tid_w / k;

    int j, r_y, sj, ej;
    Type val_a, val_x;

    for (int i = start + wid; i < end; i += num_warps) { // warp takes the row
        r_y = i * k;
        sj = irp[i];
        ej = irp[i+1];

        LDS[tid_b] = 0.0;
        for (j = sj + swid; j < ej; j += sub_warps) { // sub warp takes the non-zero
            val_a = as[j];
            val_x = x[ja[j] * k + tid_sw]; // thread in sub warp takes the specific value of x

            LDS[tid_b] += val_a * val_x;
        }
        __syncwarp();

        LDS[tid_b] = sub_reduce(FULL_WARP_MASK, warpSize>>1, k, LDS[tid_b]);

        if (swid == 0) y[r_y + tid_sw] = LDS[tid_b];
        __syncwarp();
    }
}

/**
  * VECTOR KERNEL with k < warpSize and warpSize non multiple of k: 1 warp per matrix row
  * */
__device__ void spmm_csr_vector_small_uneven(const int *irp, const int *ja, const Type *as, int start, int end,
                                      int k, const Type* x, Type* y){

    extern __shared__ Type LDS[];

    int tid_b = threadIdx.x, tid_w = tid_b % warpSize, tid_sw = tid_b % k;
    int num_warps = blockDim.x / warpSize, sub_warps = ROUND_UP(warpSize,k);
    int wid = tid_b / warpSize, swid = tid_w / k;

    // reduction prep
    int rem = warpSize % k, base = (sub_warps * (swid + 1)) - 1;
    unsigned mask;
    int s, first_pot;
    //TODO: optimize this setting --> see if it's better to compute in csr_adaptive kernel
    if (sub_warps >> 3) {
        first_pot = 8;
    } else if (sub_warps >> 2) {
        first_pot = 4;
    } else {
        first_pot = 2;
    }
    s = first_pot * k;
    mask = __ballot_sync(FULL_WARP_MASK, tid_w < s);
    int start_c;
    int residual_warps = sub_warps - first_pot, z = first_pot + swid;

    int j, r_y, sj, ej;
    Type val_a, val_x;
    for (int i = start + wid; i < end; i += num_warps) { // warp takes the row
        r_y = i * k;
        sj = irp[i];
        ej = irp[i+1];

        LDS[tid_b] = 0.0;
        for (j = sj + swid; j < ej; j += sub_warps) { // sub warp takes the non-zero
            val_a = as[j];
            val_x = x[ja[j] * k + tid_sw]; // thread in sub warp takes the specific value of x

            LDS[tid_b] += val_a * val_x;
        }

        // TODO: reduce thread divergence using other conditions than 'uneven' boolean
        // accumulate the excluded values due to the unevenness of the sub warps size
        j = base + sj;
        start_c = rem - (i % k);
        if (tid_sw == start_c && j < ej) {
            val_a = as[j];
            val_x = x[ja[j] * k + tid_sw];

            LDS[tid_b] += val_a * val_x;
        }
        __syncwarp();

        // first reduction
        if (swid < residual_warps) {
            if ((z != sub_warps - 1) || ()){
                LDS[tid_b] += LDS[tid_b + z * k];
            }
        }
        __syncwarp();

        if (swid < first_pot) LDS[tid_b] = sub_reduce(mask, s>>1, k, LDS[tid_b]);
        if (swid == 0) y[r_y + tid_sw] = LDS[tid_b];
        __syncwarp();
    }
}

/**
 * STREAM KERNEL: streaming into the local scratchpad memory of a fixed number of non-zeros to assign each warp
 * - Becomes inoperative if a row has more NZs than can be allocated in the scratchpad. --> TODO
 * */
__device__ void spmm_csr_stream(const int *irp, const int *ja, const Type *as, int row_start, int row_end,
                                int k, const Type* x, Type* y){

    int i, j;
    int first_nz = irp[row_start];
    int tot_nz = irp[row_end] - first_nz;

    extern __shared__ Type LDS[]; // it must be reused for every column of x to avoid overflowing available memory

    int tid_b = threadIdx.x;
    int thread_nz = first_nz + tid_b;

    Type tmp, val = as[thread_nz];
    int r_x = ja[thread_nz] * k;

    for (int z = 0; z < k; z++) {
        // stream the first iteration of SpMM into LDS
        if (tid_b < tot_nz) LDS[tid_b] = val * x[r_x + z];
        __syncthreads();

        // Linear reduction: sum up the partial results --> may leave some threads idle
        for (i = row_start + tid_b; i < row_end; i += blockDim.x){
            tmp = 0.0;
            for (j = (irp[i]-first_nz); j < (irp[i + 1]-first_nz); j++){
                tmp += LDS[j];
            }
            y[i*k + z] = tmp;
        }
        __syncthreads();
    }
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

    if (k >= warpSize) {
        spmm_csr_vector_large(irp, ja, as, block_row_start, block_row_end, k, x, y);
    } else if (warpSize % k) {
        spmm_csr_vector_small_uneven(irp, ja, as, block_row_start, block_row_end, k, x, y);
    } else {
        spmm_csr_vector_small_even(irp, ja, as, block_row_start, block_row_end, k, x, y);
    }

    /*
    if (rows > MAX_NUM_ROWS) { // the rows are not so long: non-zeros can fit into the LDS
        spmm_csr_stream(irp, ja, as, block_row_start, block_row_end, k, x, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        if (k >= warpSize) {
            spmm_csr_vector_large(irp, ja, as, block_row_start, block_row_end, k, x, y);
        } else {
            spmm_csr_vector_small(irp, ja, as, block_row_start, block_row_end, k, x, y);
        }
    }
     */
}

/**
 * Computes the number of rows to give each block (s.t. # NZ <= BD) and the total number of blocks to cover all rows.
 * So that NZs can fit into LDS entries of size BDx.
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

    int bd, gd;

    // 1D block dimension --> 512 seems feasible for biggest matrices
    int avg = ROUND_UP(nz,m);
    bd = ROUND_UP_MULT(avg,WARP_SIZE);
    if (bd > MAX_THREADS_BLOCK) bd = MAX_THREADS_BLOCK;
    *BLOCK_DIM = dim3(bd);

    // compute row balancing on blocks
    gd = get_csr_row_blocks(bd, m, irp, blocks);
    *num_blocks = gd;

    // compute shared memory dimension
    int k_factor = ROUND_UP(WARP_SIZE,k);
    *shared_mem = bd * k_factor * sizeof(Type);

    // set block grid dimension to spawn 'num_blocks' blocks
    *GRID_DIM = dim3(gd-1);

    printf("GRID(%d) - BLOCK(%d) - SHM(%d)\n", gd-1, bd, *shared_mem);
}