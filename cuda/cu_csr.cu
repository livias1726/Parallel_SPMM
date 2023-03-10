#include "headers/cu_csr.cuh"

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
                                int k, const Type* x, int sm_dim, Type* y){

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
 * - Coalesced loads
 * - Efficient utilization of the GPU's DRAM bandwidth
 *
 * Problems:
 *      - Loses efficiency when a warp operates on rows with a large number of NZs. --> vector kernel
 *      - Becomes inoperative if a row has more NZs than can be allocated in the scratchpad. --> TODO
 *
 * Inspired by Algorithm 3 of
 * 'Greathouse, Daga - Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format'
 * */
__device__ void spmm_csr_stream(const int *irp, const int *ja, const Type *as, int row_start, int row_end,
                                int k, const Type* x, int sm_dim, Type* y){

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
__global__ void spmm_csr_adaptive_kernel(const int *irp, const int *ja, const Type *as, int k,
                                         const Type* x, int* blocks, int sm_dim, Type* y) {

    int block_row_start = blocks[blockIdx.x];
    int block_row_end = blocks[blockIdx.x + 1];
    int rows = block_row_end - block_row_start;

    if (rows > MAX_NUM_ROWS) { // the rows are not so long: non-zeros can fit into the LDS
        spmm_csr_stream(irp, ja, as, block_row_start, block_row_end, k, x, sm_dim, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        spmm_csr_vector(irp, ja, as, block_row_start, block_row_end, k, x, sm_dim, y);
    }
}

void compute_csr_dimensions(int m, int k, int *irp, int* blocks, int *num_blocks, dim3* BLOCK_DIM, dim3* GRID_DIM,
                            int *shared_mem){

    int max_nz;
    *num_blocks = get_csr_row_blocks(m, irp, blocks, &max_nz);

    // compute shared memory dimension
    *shared_mem = get_shared_memory(max_nz, k);
    if (*shared_mem == -1) {
        printf("TOO MANY NZ\n");
        cudaDeviceReset();
    }
    *BLOCK_DIM = dim3(BD);
    *GRID_DIM = dim3(*num_blocks-1);
}