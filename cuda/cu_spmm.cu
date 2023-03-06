#include "cu_utils.cuh"

__device__ double warp_reduce(double sum){
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
__device__ void product_csr_vector(const int *irp, const int *ja, const double *as, int start, int end,
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
__device__ void product_csr_stream(const int *irp, const int *ja, const double *as, int row_start, int row_end,
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
__global__ void product_csr_adaptive(const int *irp, const int *ja, const double *as, int k,
                                     const double* x, int* blocks, int sm_dim, double* y) {

    int block_row_start = blocks[blockIdx.x];
    int block_row_end = blocks[blockIdx.x + 1];
    int rows = block_row_end - block_row_start;

    if (rows > MAX_NUM_ROWS) { // the rows are not so long: non-zeros can fit into the LDS
        product_csr_stream(irp, ja, as, block_row_start, block_row_end, k, x, sm_dim, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        product_csr_vector(irp, ja, as, block_row_start, block_row_end, k, x, sm_dim, y);
    }
}

int main(int argc, char** argv) {
    // host
    MM_typecode t;
    FILE *f;
    CSR *csr;
    int k, m, n, nz;
    double gflops_s, gflops_p, abs_err, rel_err;
    double *x, *y_s, *y_p, flop;
    // device
    double *d_x, *d_y, *d_as;
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
    #ifdef DEBUG
    print_ell(ell);
    #endif
    //TODO: allocCudaEll()
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

#ifdef ELLPACK
    //TODO
#else

     // ADAPTIVE
    int *d_blocks, *blocks, num_blocks, max_nz;

    // compute #rows per block
    blocks = (int*)malloc(m*sizeof(int));
    //malloc_handler(1, (void*[]){blocks});
    num_blocks = get_csr_row_blocks(m, csr->IRP, blocks, &max_nz);
    checkCudaErrors(cudaMalloc((void**) &d_blocks, num_blocks*sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_blocks, blocks, num_blocks*sizeof(int), cudaMemcpyHostToDevice));

    // compute shared memory dimension
    const int shared_mem = get_shared_memory(max_nz, k);
    if (shared_mem == -1) {
        printf("TOO MANY NZ\n");
        cudaDeviceReset();
    }
    const dim3 BLOCK_DIM = dim3(BDX);
    const dim3 GRID_DIM = dim3(num_blocks-1);

    // product
    timer->start();
    product_csr_adaptive<<<GRID_DIM, BLOCK_DIM, shared_mem>>>(d_irp, d_ja, d_as, k, d_x, d_blocks, shared_mem/2, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
#endif

    gflops_p = (double)flop/((timer->getTime())*1.e6);
    checkCudaErrors(cudaMemcpy(y_p, d_y, m * k * sizeof(double), cudaMemcpyDeviceToHost));

    // check results
    // --> relative error should be as close as possible to 2.22e−16 (IEEE double precision unit roundoff)
    get_errors(m, k, y_s, y_p, &abs_err, &rel_err);

// ------------------------------- Cleaning up ------------------------------ //
    delete timer;

#ifdef ELLPACK
    //TODO
    delete[] ell;
#else
    checkCudaErrors(cudaFree(d_irp));
    checkCudaErrors(cudaFree(d_ja));
    checkCudaErrors(cudaFree(d_as));
    delete[] csr;
#endif

    checkCudaErrors(cudaFree(d_blocks));
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