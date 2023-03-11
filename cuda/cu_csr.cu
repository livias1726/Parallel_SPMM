#include "headers/cu_csr.cuh"

//TODO: check error accumulation
__device__ Type warp_reduce(Type sum){
    // implementation of a logarithmic reduction with warp-level communication primitive
    for(int s = warpSize >> 1; s > 0; s >>= 1) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
    }
    return sum;
}

/**
  * VECTOR KERNEL: 1 warp per matrix row
  *     - Coalesced memory accesses (labeled by warp index) to JA and AS, followed by a reduction phase.
  *
  * Problems: efficient execution demands a number of NZs per row greater than the warp size.
  *
  * Note: if rows don't have more than 'warpSize' NZs each, no warp iterates more than once on the CSR arrays.
  *       Else, the order of summation differs from the scalar kernel (error accumulation).
  * */
__device__ void spmm_csr_vector(const int *irp, const int *ja, const Type *as, int start, int end,
                                int k, const Type* x, Type* y){

    // use of shared memory
    extern __shared__ Type LDS[]; // used to write temporary product results

    int tx = threadIdx.x;
    int twid = tx & (warpSize-1); // thread index within the warp
    int wid = tx/warpSize; // warp index within the block
    int num_warps = blockDim.x/warpSize; // number of warps in block

    for (int i = start+wid; i < end; i += num_warps) { // each warp in the block is given a row
        int s_idx = irp[i];
        int e_idx = irp[i+1];

        for (int z = 0; z < k; z++) {
            // compute partial products
            LDS[tx] = 0.0;
            for (int j = s_idx + twid; j < e_idx; j += warpSize) { // each thread in the warp operates on a single nz of the row
                LDS[tx] += as[j] * x[ja[j] * k + z];
            }
            __syncthreads();

            LDS[tx] = warp_reduce(LDS[tx]);

            // first thread writes warp result
            if (twid == 0) y[i * k + z] = LDS[tx];
            __syncthreads();
        }
    }
}

/**
 * STREAM KERNEL: streaming into the local scratchpad memory of a fixed number of non-zeros to assign each warp
 *
 * Problems:
 *      - Loses efficiency when a warp operates on rows with a large number of NZs. --> vector kernel
 *      - Becomes inoperative if a row has more NZs than can be allocated in the scratchpad. --> TODO
 *
 * Inspired by Algorithm 3 of Greathouse and Daga's
 * "Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format"
 * */
__device__ void spmm_csr_stream(const int *irp, const int *ja, const Type *as, int row_start, int row_end,
                                int k, const Type* x, Type* y){

    int i;
    int first_nz = irp[row_start];
    int tot_nz = irp[row_end] - first_nz;

    extern __shared__ Type LDS[]; // it must be reused for every column of x to avoid overflowing available memory

    int tid_block = threadIdx.x;
    int thread_nz = first_nz + tid_block;

    for (int z = 0; z < k; z++) {
        // stream the first iteration of SpMM into LDS using l_tid to shift on the values
        if (tid_block < tot_nz) LDS[tid_block] = as[thread_nz] * x[ja[thread_nz]*k + z]; // efficient bandwidth usage
        __syncthreads();

        // Linear reduction: sum up the partial results --> may leave some threads idle
        for (i = row_start + tid_block; i < row_end; i += blockDim.x){
            double temp = 0.0;

            for (int j = (irp[i]-first_nz); j < (irp[i + 1]-first_nz); j++){
                temp += LDS[j];
            }

            y[i*k + z] = temp;
        }
        __syncthreads();
    }
}


/**
 * CSR Adaptive SpMM
 *
 * Dynamically determines whether to execute a set of rows with the stream or the vector kernel.
 * */
__global__ void spmm_csr_adaptive_kernel(const int *irp, const int *ja, const Type *as, int k, const Type* x,
                                         int* blocks, Type* y) {

    int block_row_start = blocks[blockIdx.x];
    int block_row_end = blocks[blockIdx.x + 1];
    int rows = block_row_end - block_row_start;

    if (rows > MAX_NUM_ROWS) { // the rows are not so long: non-zeros can fit into the LDS
        spmm_csr_stream(irp, ja, as, block_row_start, block_row_end, k, x, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        spmm_csr_vector(irp, ja, as, block_row_start, block_row_end, k, x, y);
    }
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

    // 1D block dimension
    //int bd = nz + nz % WARP_SIZE; // first multiple of warp bigger than nz
    int avg = GET_SUP_INT(nz,m) int bd = avg + WARP_SIZE - avg%WARP_SIZE;
    if (bd > MAX_THREADS_BLOCK) bd = MAX_THREADS_BLOCK;
    *BLOCK_DIM = dim3(bd);

    // compute row balancing on blocks
    *num_blocks = get_csr_row_blocks(bd, m, irp, blocks);

    // compute shared memory dimension
    *shared_mem = bd * sizeof(Type); //get_shared_memory(bd);

    // set block grid dimension to spawn 'num_blocks' blocks
    *GRID_DIM = dim3(*num_blocks-1);
}