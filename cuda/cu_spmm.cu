#include "cu_utils.cuh"

// -------------------------------------------- GPU Utils --------------------------------------------------- //
__device__ double warp_reduce(double sum){
    // implementation of a logarithmic reduction with warp-level communication primitive
    for(int s = warpSize >> 1; s > 0; s >>= 1) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
    }

    return sum;
}

// --------------------------------------------- SpMM ----------------------------------------------------//
/**
  * VECTOR KERNEL: 1 warp per matrix row
  *     - Coalesced memory accesses (labeled by warp index) to JA and AS, followed by a reduction phase.
  *
  * Problems: efficient execution demands a number of NZs per row greater than the warp size.
  *
  * Note: if rows don't have more than 'warpSize' NZs each, no warp iterates more than once on the CSR arrays.
  *       Else, the order of summation differs from the scalar kernel (error accumulation).
  * */
__device__ void spmm_csr_vector(const int *irp, const int *ja, const double *as, int start, int end,
                                   int k, const double* x, int sm_dim, double* y){

    // use of shared memory
    extern __shared__ double LDS[]; // used to write temporary product results

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
__device__ void spmm_csr_stream(const int *irp, const int *ja, const double *as, int row_start, int row_end,
                                   int k, const double* x, int sm_dim, double* y){

    int i;
    int first_nz = irp[row_start];
    int tot_nz = irp[row_end] - first_nz;

    extern __shared__ double LDS[]; // it must be reused for every column of x to avoid overflowing available memory

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
__global__ void spmm_csr_adaptive_kernel(const int *irp, const int *ja, const double *as, int k,
                                     const double* x, int* blocks, int sm_dim, double* y) {

    int block_row_start = blocks[blockIdx.x];
    int block_row_end = blocks[blockIdx.x + 1];
    int rows = block_row_end - block_row_start;

    if (rows > MAX_NUM_ROWS) { // the rows are not so long: non-zeros can fit into the LDS
        spmm_csr_stream(irp, ja, as, block_row_start, block_row_end, k, x, sm_dim, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        spmm_csr_vector(irp, ja, as, block_row_start, block_row_end, k, x, sm_dim, y);
    }
}

/**
 * Ellpack SpMM kernel
 *
 * Each block is given a subset of rows in A (using AS and JA) and a subset of columns of x.
 * To completely cover A's rows, blocks will eventually need to iterate horizontally on AS and JA and accumulate products.
 *
 * Each thread in a block is responsible for a single non-zero in A and uses x values
 * (in the specific block stored in shared memory) to compute the partial product to accumulate on the row -->
 * */

 // TODO: version 1 --> customized test
__global__ void spmm_ell_kernel_1(unsigned int rows, unsigned int maxnz, const int *ja, const Type *as, const Type *x,
                                  unsigned int k, Type* y){

    const int bdx = blockDim.x, bdy = blockDim.y;

    // blocks with the same bx treats the same rows of A
    const int bx = blockIdx.x; // gives the row block of A --> as and ja
    // blocks with the same by treats the same columns of x
    const int z = blockIdx.y; // gives the column of x the block is responsible for --> y[][z]

    //blockDim.x * (blockIdx.x + blockIdx.y*gridDim.x) + threadIdx.x;
    const int i = threadIdx.x + (bdx * bx); // global row of the thread
    if (i >= rows) return;
    const int col = threadIdx.y; // thread's column in AS and JA --> to retrieve correct column idx

    //printf("BX: %d, BY: %d, TX: %d, TY: %d, GlobalTX: %d\n", bx, z, threadIdx.x, col, i);

    Type nz_val, temp = 0.0;

    int idx, j;
    for (int jj = col; jj < maxnz; jj += bdy) {
        idx = i*maxnz + jj;

        nz_val = as[idx];
        if (nz_val == 0) break;

        j = ja[idx];
        temp += __dmul_rn(nz_val, x[j*k + z]);
    }
    __syncthreads();

    y[i*k + z] = temp;
}

// TODO: version 2 --> CUDA Programming Guide
/*
__global__ void spmm_ell_kernel_2(const int rows, const int cols, const int maxnz, const int stride, const int *ja,
                                  const double *as, const double *x, const int k, double* y) {

    int blockRow = blockIdx.y, blockCol = blockIdx.x;

    // each thread block computes one sub-matrix of y
    Matrix l_y = GetSubMatrix(y, blockRow, blockCol);

    // each thread computes one element of l_y by accumulating results into temp
    Type temp = 0.0;

    // thread row and column within l_y
    int row = threadIdx.y, col = threadIdx.x;

    // load sub-matrix of x
    l_x;

    // loop over all the sub-matrices of as/ja and x that are required to compute l_y
    // multiply each pair of sub-matrices together and accumulate the results
    for (int m = 0; m < (maxnz / BD); ++m) {
        // Get sub-matrix Asub of A
        Matrix l_as = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
 */

// TODO: version 3 --> spmv for ellpack
/*
__global__ void spmm_ell_kernel_3(const int rows, const int cols, const int maxnz, const int stride, const int *ja,
                                  const Type *as, const Type *x, const int k, Type* y){

    int i = blockDim.x * (blockIdx.x + blockIdx.y*gridDim.x) + threadIdx.x;

    if (i >= rows) return;

    Type temp = 0.0;

    // shift to thread base in JA and AS
    ja += i;
    as += i;

    for (int j = 0; j < maxnz; j++) {
        int pointer = ja[0] - firstIndex;
        double value = as[0];

        temp += __dmul_rn(value, x[pointer]);

        ja += pitch;
        as += pitch;
    }

    y[i] = temp;
}
 */

int main(int argc, char** argv) {
    // host
    MM_typecode t;
    FILE *f;
    CSR *csr;
    ELL *ell;
    int k, m, n, nz, maxnz;
    double flop, gflops_s, gflops_p, abs_err, rel_err;
    Type *x, *y_s, *y_p;
    // device
    Type *d_x, *d_y, *d_as;
    int *d_irp, *d_ja;

    // ----------------------- Set Up ------------------------------------------- //

    process_arguments(argc, argv, &f, &k);
    process_mm(&t, f);

    // read matrix from file
    Elem** elems = read_mm(f, &m, &n, &nz, t);
    fclose(f);

    // ----------------------- Host memory initialisation ----------------------- //

    // convert to wanted storage format
#ifdef ELLPACK
    //TODO: manage H-Ellpack
    ell = read_mm_ell(elems, m, n, nz);
    maxnz = ell->MAXNZ;
    #ifdef DEBUG
        print_ell(ell);
    #endif

    alloc_cuda_ell(ell, &d_ja, &d_as);
#else
    csr = read_mm_csr(elems, m, n, nz);
    #ifdef DEBUG
        print_csr(csr);
    #endif

    alloc_cuda_csr(csr, &d_irp, &d_ja, &d_as);
#endif

    flop = (double)2*k*nz;

    alloc_struct(&x, n, k);
    alloc_struct(&y_s, m ,k);
    alloc_struct(&y_p, m ,k);

    populate_multivector(x, n, k);

#ifdef DEBUG
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

    alloc_cuda_spmm(&d_x, &d_y, x, m, n, k);

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    // ------------------------------------------- Serial CPU SpMM --------------------------------------------- //
    timer->start();
#ifdef ELLPACK
    serial_product_ell(ell, x, k, y_s);
#else
    serial_product_csr(csr, x, k, y_s);
#endif
    timer->stop();

    gflops_s = (double)flop/((timer->getTime())*1.e6);
    timer->reset();
    // --------------------------------------------- GPU SpMM -------------------------------------------------- //
    // pre-processing
    checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)); //to avoid bank conflicts since double values are used

    // Compute BLOCK_DIM --> each block works on a sub-matrix of A (bdy x n) and a sub-matrix of x (n x bdx)
    dim3 BLOCK_DIM;
    dim3 GRID_DIM;
    int shared_mem;

#ifdef ELLPACK
    compute_ell_dimensions(m, maxnz, k, &BLOCK_DIM, &GRID_DIM, &shared_mem);

    // product
    timer->start();
    spmm_ell_kernel_1<<<GRID_DIM, BLOCK_DIM,shared_mem>>>(m, maxnz, d_ja, d_as, d_x, k, d_y);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
#else
    // ADAPTIVE
    int *d_blocks, *blocks, num_blocks, max_nz;
    blocks = (int*)malloc(m*sizeof(int));
    malloc_handler(1, (void*[]){blocks});

    compute_csr_dimensions(m, k, csr->IRP, blocks, &num_blocks, &BLOCK_DIM, &GRID_DIM, &shared_mem);
    checkCudaErrors(cudaMalloc((void**) &d_blocks, num_blocks*sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_blocks, blocks, num_blocks*sizeof(int), cudaMemcpyHostToDevice));

    // product
    timer->start();
    spmm_csr_adaptive_kernel<<<GRID_DIM, BLOCK_DIM, shared_mem>>>(d_irp, d_ja, d_as, k, d_x, d_blocks, shared_mem/2, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();

#endif

    gflops_p = (double)flop/((timer->getTime())*1.e6);
    checkCudaErrors(cudaMemcpy(y_p, d_y, m * k * sizeof(double), cudaMemcpyDeviceToHost));

    // check results
    // --> relative error should be as close as possible to 2.22e−16 (IEEE double precision unit roundoff)
    get_errors(m, k, y_s, y_p, &abs_err, &rel_err);

#ifdef SAVE
    save_result(y_p, m, k);
#endif
#ifdef DEBUG
    print_matrix(y_s, m, k, "\nSerial Result:\n");
    print_matrix(y_p, m, k, "\nParallel Result:\n");
#endif

// ------------------------------- Cleaning up ------------------------------ //
    delete timer;

#ifdef ELLPACK
    //TODO
    delete[] ell;
#else
    checkCudaErrors(cudaFree(d_irp));
    checkCudaErrors(cudaFree(d_ja));
    checkCudaErrors(cudaFree(d_as));
    checkCudaErrors(cudaFree(d_blocks));
    delete[] csr;
#endif

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    delete[] x;
    delete[] y_s;
    delete[] y_p;

    cudaDeviceReset();

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %f %f", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "Serial GFLOPS: %f\nParallel GFLOPS: %f\nAbsolute error: %.2e\nRelative error: %.2e\n",
            gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
}