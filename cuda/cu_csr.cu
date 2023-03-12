#include "headers/cu_csr.cuh"

__device__ Type warp_reduce(Type sum){
    // implementation of a logarithmic reduction with warp-level communication primitive
    for(int s = warpSize >> 1; s > 0; s >>= 1) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
    }
    return sum;
}

__device__ Type sub_reduce(unsigned mask, int size, Type sum){
    // implementation of a logarithmic reduction with warp-level communication primitive
    for(int s = size >> 1; s > 0; s >>= 1) {
        sum += __shfl_down_sync(mask, sum, s);
    }
    return sum;
}

/**
  * VECTOR KERNEL: 1 warp per matrix row.
  * */
  //TODO: manage case k > warpSize
  // --> increase LDS by a factor of ceil(k/warpSize)
  // ----> with bd=MAX, needed a k x7 bigger (> 192) than warpSize to reach over MAX_SHM
  // --> each thread in the 'sub_warp' uses ceil(k/warpSize) consecutive cells in LDS to store accumulation
  // --> reduction needs to take into account this stride (ceil(k/warpSize))
__device__ void spmm_csr_vector(const int *irp, const int *ja, const Type *as, int start, int end,
                                int k, const Type* x, Type* y){

    // use of shared memory
    extern __shared__ Type LDS[]; // used to write temporary product results

    int tid_b = threadIdx.x;
    int tid_w = tid_b % warpSize;

    int sub_warps = warpSize/k;
    int swid = tid_w / k; // sub warp id
    int tid_sw = tid_b % k; // thread index within the sub warp

    int wid = tid_b / warpSize; // warp index within the block
    int num_warps = blockDim.x / warpSize; // number of warps in block

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

        // reduction
        /*
        for (z = 0; z < k; z++) {
            unsigned mask = __ballot_sync(FULL_WARP_MASK, tid_sw == z);
            LDS[tid_b] = sub_reduce(mask, sub_warps, LDS[tid_b]);
        }
         */
        if (swid == 0) {
            for (int sw = 1; sw < sub_warps; sw++) {
                LDS[tid_b] += LDS[tid_b + k * sw];
            }
        }
        __syncwarp();

        // first thread writes result
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

    spmm_csr_vector(irp, ja, as, block_row_start, block_row_end, k, x, y);
    /*
    if (rows > MAX_NUM_ROWS) { // the rows are not so long: non-zeros can fit into the LDS
        spmm_csr_stream(irp, ja, as, block_row_start, block_row_end, k, x, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        spmm_csr_vector(irp, ja, as, block_row_start, block_row_end, k, x, y);
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

/**
 * Computes the dimension of the dynamic shared memory to be used in each block.
 * */
int get_shared_memory(int bd){
    int shm = bd*sizeof(Type); // one nz per thread
    return shm;
}

void compute_csr_dimensions(int m, int nz, int k, int *irp, int* blocks, int *num_blocks, dim3* BLOCK_DIM, dim3* GRID_DIM,
                            int *shared_mem){

    int bd, gd;

    // 1D block dimension --> 512 seems feasible for biggest matrices
    int avg = nz/m;
    bd = avg + WARP_SIZE - avg%WARP_SIZE; //bd = nz + WARP_SIZE - nz%WARP_SIZE;
    if (bd > MAX_THREADS_BLOCK) bd = MAX_THREADS_BLOCK;
    *BLOCK_DIM = dim3(bd);

    // compute row balancing on blocks
    gd = get_csr_row_blocks(bd, m, irp, blocks);
    *num_blocks = gd;

    // compute shared memory dimension
    *shared_mem = bd * sizeof(Type); //get_shared_memory(bd);

    // set block grid dimension to spawn 'num_blocks' blocks
    *GRID_DIM = dim3(gd-1);

    printf("GRID(%d) - BLOCK(%d) - SHM(%d)\n", gd-1, bd, *shared_mem);
}