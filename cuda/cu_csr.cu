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
 *
 * NOTE: using __dmul_rn to perform the accumulation guarantees 0.0 errors, but halves the performances!
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
 *
 * NOTE: using __dmul_rn to perform the accumulation does not reduce the error and decreases the performance!
 * */
__device__ void spmm_csr_vector_small(const int *irp, const int *ja, const Type *as, int start, int end,
                                      int k, const Type* x, Type* y){

    extern __shared__ Type LDS[];           // each thread has a cell in the shared memory

    int tid_b = threadIdx.x;                // thread id in the block
    int tid_w = tid_b % warpSize;           // thread id in the warp
    int tid_sw = tid_w % k;                 // thread id in the sub warp
    int wid = tid_b / warpSize;             // warp id in the block
    int swid = tid_w / k;                   // sub warp id in the warp
    int warps = blockDim.x / warpSize;      // number of warps in the block
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

    int j, r_y;
    for (int i = start + wid; i < end; i += warps) { // warp takes the row
        r_y = i * k;

        // ACCUMULATION
        LDS[tid_b] = 0.0;
        if (swid != sub_warps) { // excludes the truncated sub-warp
            for (j = irp[i] + swid; j < irp[i+1]; j += sub_warps) { // sub warp takes the non-zero
                // thread in sub warp takes the specific value of x
                LDS[tid_b] += as[j] * x[ja[j] * k + tid_sw];
            }
        }

        /*
         * REDUCTION 1: values of sub-warps not involved in the warp reduction phase are accumulated
         * in parallel by the other sub-warps.
         * */
        if (swid < excluded) LDS[tid_b] += LDS[tid_b + s];

        // REDUCTION 2
        if (swid < first_pot) LDS[tid_b] = sub_reduce(mask, s>>1, k, LDS[tid_b]);

        // update
        if (swid == 0) y[r_y + tid_sw] = LDS[tid_b];
    }
}

/**
 * VECTOR KERNEL: 1 warp per matrix row.
 *      If k is higher than half the warp size, the spmm with sub-warps (spmm_csr_vector_small)
 *      behaves just like the other (spmm_csr_vector_large) but with more useless overhead.
 * */
__global__ void spmm_csr_vector_kernel(const int *irp, const int *ja, const Type *as, int k, const Type* x,
                                         int* blocks, Type* y) {

    int start = blocks[blockIdx.x];
    int end = blocks[blockIdx.x + 1];

    if (k >= warpSize >> 1) {
        spmm_csr_vector_large(irp, ja, as, start, end, k, x, y);
    } else {
        spmm_csr_vector_small(irp, ja, as, start, end, k, x, y);
    }
}

/**
 * Computes the number of rows to give each block and the total number of blocks to cover all rows.
 *
 * Inspired by Algorithm 2 of Greathouse, Daga - "Efficient Sparse Matrix-Vector Multiplication
 * on GPUs using the CSR Storage Format"
 *
 * @param bd                block dimension
 * @param rows              total number of rows
 * @param irp               IRP array of the CSR format
 * @param rows_per_block    output array with the starting row per block
 * */
int get_rows_per_block(int bd, int rows, int* irp, int* rows_per_block){

    int nz = 0, last_i = 0, ctr = 1;
    rows_per_block[0] = 0;

    for (int i = 1; i < rows; i++) {
        nz += irp[i] - irp[i-1]; // count the sum of non-zeros in the considered rows

        // the block can process more non-zeros
        if (nz < bd) continue;

        // there are more non-zeros than threads in a block AND
        // more than max rows were scanned for the block: decrease number of rows for the block
        if ((nz > bd) && (i - last_i > 1)) --i;

        // update rows_per_block
        last_i = i;
        rows_per_block[ctr++] = i;
        nz = 0;
    }

    rows_per_block[ctr++] = rows;
    return ctr;
}

/**
 * Compute the dimensions of the kernel w.r.t. the number of non-zeros per row and k.
 *
 * @param csr           CSR structure
 * @param k             number of columns in the multi-vector
 * @param blocks        array of pointers to starting row per block
 * @param num_blocks    total number of blocks computed
 * @param BLOCK_DIM     pointer to the block dimensions
 * @param GRID_DIM      pointer to the grid dimensions
 * @param shared_mem    pointer to the amount of shared memory
 * */
void compute_csr_dimensions(CSR* csr, int k, int* blocks, int *num_blocks, dim3* BLOCK_DIM, dim3* GRID_DIM, int *shared_mem){

    int m = csr->M, nz = csr->NZ, *irp = csr->IRP;

    // 1D BLOCKS: average number of non-zeros per row rounded up to a multiple of warpSize
    int bd = ROUND_UP_MULT(ROUND_UP(nz,m), WARP_SIZE);
    if (bd > MAX_THREADS_BLOCK) bd = MAX_THREADS_BLOCK; // check maximum threads per block limit

    // shared memory is used only when k is lower than half warp size (spmm_csr_vector_small)
    // no need to check memory limit for the limit on BD will limit the memory to 1024 * 8
    int shm = (k < WARP_SIZE >> 1) ? bd * sizeof(Type) : 0;

    // 1D GRID: compute row balancing on blocks and number of blocks needed
    int nb = get_rows_per_block(bd, m, irp, blocks);

    // output
    *BLOCK_DIM = dim3(bd);
    *shared_mem = shm;
    *num_blocks = nb;
    *GRID_DIM = dim3(nb-1);
}

void alloc_cuda_csr(CSR* csr, int* blocks, int num_blocks, int **d_irp, int **d_ja, Type **d_as, int **d_blocks){
    int m = csr->M, nz = csr->NZ;
    int *irp = csr->IRP, *ja = csr->JA;
    Type *as = csr->AS;

    int size_irp = (m+1)*sizeof(int);
    int size_ja = nz*sizeof(int);
    int size_as = nz*sizeof(Type);
    int size_blocks = num_blocks*sizeof(int);

    printf("CSR BYTES: %d\n", size_irp + size_ja + size_as + size_blocks);

    checkCudaErrors(cudaMalloc((void**) d_irp, size_irp));
    checkCudaErrors(cudaMalloc((void**) d_ja, size_ja));
    checkCudaErrors(cudaMalloc((void**) d_as, size_as));
    checkCudaErrors(cudaMalloc((void**) d_blocks, size_blocks));

    checkCudaErrors(cudaMemcpy(*d_irp, irp, size_irp, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_ja, ja, size_ja, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_as, as, size_as, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_blocks, blocks, size_blocks, cudaMemcpyHostToDevice));
}