#include "cu_utils.cuh"

/**
  * VECTOR KERNEL: 1 warp per matrix row
  *     - Coalesced memory accesses (labeled by warp index) to JA and AS, followed by a reduction phase.
  *
  * Problems: efficient execution demands a number of NZs per row greater than the warp size.
  *
  * Note: if rows don't have more than 'warpSize' NZs each, no warp iterates more than once on the CSR arrays.
  *       Else, the order of summation differs from the scalar kernel (error accumulation).
  * */
__device__ void product_csr_vector(const int *irp, const int *ja, const double *as, int start, int end, int l_tid,
                                   int k, const double* x, double* y){

    int rows = end - start;

    // use of shared memory
    __shared__ double LDS[BD][MAX_K]; // used to write temporary product results
    __shared__ int shared_ptrs[MAX_NUM_ROWS][2]; // used to share start and end irp pointers

    int tid = l_tid + (blockIdx.x * blockDim.x); // global thread index
    int thread_lane = l_tid & (warpSize-1); // thread index within the warp
    int wid = tid/warpSize; // global warp index
    int warp_lane = l_tid/warpSize; // warp index within the block
    int num_warps = (blockDim.x/warpSize) * gridDim.x; // total number of active warps

    for(int i = wid; i < rows; i += num_warps){ // rows are shared among warps
        // TODO using two threads to fetch irp values is faster
        if(thread_lane < 2) shared_ptrs[warp_lane][thread_lane] = irp[i + thread_lane];

        int s_idx = shared_ptrs[warp_lane][0]; //row_start = irp[i];
        int e_idx = shared_ptrs[warp_lane][1]; //row_end = irp[i+1];

        int z;

        // compute partial products
        for (z = 0; z < k; z++) {
            LDS[l_tid][z] = 0.0;
            for (int j = s_idx + thread_lane; j < e_idx; j += warpSize) {
                LDS[l_tid][z] += as[j] * x[ja[j] * k + z];
            }
        }

        // implementation of a logarithmic reduction (V3)
        for (z = 0; z < k; z++) {
            for(unsigned int s = warpSize>>1; s > 0; s >>= 1) {
                if (l_tid < s) {
                    LDS[l_tid][z] += LDS[l_tid + s][z];
                }
            }
            __syncwarp();
        }

        // first thread writes warp result
        if (thread_lane == 0) {
            for (z = 0; z < k; z++) {
                y[i * k + z] = LDS[l_tid][z];
            }
        }
    }
}

/**
 * STREAM KERNEL: streaming into the local scratchpad memory of a fixed number of non-zeros to assign each warp
 *      - Coalesced loads
 *      - Efficient utilization of the GPU's DRAM bandwidth
 * Problems:
 *      - Loses efficiency when a warp operates on rows with a large number of NZs.
 *      - Becomes inoperative if a row has more NZs than can be allocated in the scratchpad.
 *
 * Implementation of Algorithm 3 of
 * 'Greathouse, Daga - Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format'
 * */
__device__ void product_csr_stream(const int *irp, const int *ja, const double *as, int start, int end, int l_tid,
                                   int k, const double* x, double* y){

    int i, j, z;
    int first_nz = irp[start];
    int tot_nz = irp[end] - first_nz;

    __shared__ volatile double LDS[BD][MAX_K];

    // stream the first iteration of SpMM into LDS using l_tid to shift on the values
    if (l_tid < tot_nz){
        for (z = 0; z < k; z++) {
            LDS[l_tid][z] = as[first_nz + l_tid] * x[ja[first_nz + l_tid]*k + z]; // efficient bandwidth usage
        }
    }
    __syncthreads();

    // Linear reduction: sum up the partial results
    // --> may leave some threads idle during reduction
    for (i = start + l_tid; i < end; i += blockDim.x){
        double temp;

        for (z = 0; z < k; z++) {
            temp = 0.0;

            for (j = (irp[i] - first_nz); j < (irp[i + 1] - first_nz); j++){
                temp += LDS[j][z];
            }

            y[i*k + z] = temp;
        }
    }
}

/**
 * CSR Adaptive SpMM
 *
 * Dynamically determines whether to execute a set of rows with the stream or the vector kernel.
 * */
__global__ void product_csr_adaptive(const int *irp, const int *ja, const double *as, int* blocks, int k,
                                     const double* x, double* y) {
    int start = blocks[blockIdx.x];
    int end = blocks[blockIdx.x + 1];

    int rows = end - start;
    int tid = threadIdx.x; // local thread index

    if (rows > MAX_NUM_ROWS) { // the rows are not so long: non-zeros can fit into the LDS
        product_csr_stream(irp, ja, as, start, end, tid, k, x, y);
    } else { // the single row is too large to fit into the LDS with a streaming algorithm
        product_csr_vector(irp, ja, as, start, end, tid, k, x, y);
    }
}

/**
 * SCALAR KERNEL: 1 thread per matrix row
 *
 * Problems: thread divergence
 *      - Un-coalesced memory accesses to JA and AS
 *      - Under-utilization of hardware resources:
 *              if the distribution of NZs per row is highly variable (follows a power law),
 *              many threads within a warp will remain idle while waiting for the thread with the longest row.
 * */
/*
__global__ void product_csr_scalar(const int rows, const int *irp, const int *ja, const double *as,
                                  const double *x, int k, double *y){

   int i = threadIdx.x + (blockIdx.x * blockDim.x); // global index of the thread --> index of the row
   if (i < rows){
       double temp;
       int row_start = irp[i];
       int row_end = irp[i+1];

       for(int z = 0; z < k; z++){
           temp = 0.0;
           for (int j = row_start; j < row_end; j++) {
               temp += as[j] * x[ja[j] * k + z];
           }
           y[i*k + z] = temp;
       }
   }
}
 */

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
    allocCudaCsr(csr, &d_irp, &d_ja, &d_as);
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

    allocCudaSpmm(&d_x, &d_y, x, m, n, k);

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
#ifdef ELLPACK
    //TODO
#else

    // SCALAR
     /*
    int num_blocks = (m + BD - 1)/BD;
    dim3 GRID_DIM;
    GET_GRID(m, GRID_DIM)

    timer->start();
    product_csr_scalar<<<GRID_DIM, BD>>>(m, d_irp, d_ja, d_as, d_x, k, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
     */

     // ADAPTIVE
    int *d_blocks, *blocks, num_blocks;
    blocks = (int*)malloc(m*sizeof(int));
    //malloc_handler(1, (void*[]){blocks});

    num_blocks = csr_adaptive_blocks(m, csr->IRP, blocks);
    checkCudaErrors(cudaMalloc((void**) &d_blocks, num_blocks*sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_blocks, blocks, num_blocks*sizeof(int), cudaMemcpyHostToDevice));

    const dim3 GRID_DIM = num_blocks-1;
    timer->start();
    product_csr_adaptive<<<GRID_DIM, BD>>>(d_irp, d_ja, d_as, d_blocks, k, d_x, d_y);
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

    //checkCudaErrors(cudaFree(d_blocks));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    delete[] x;
    delete[] y_s;
    delete[] y_p;

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %f %f", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "Serial GFLOPS: %f\nParallel GFLOPS: %f\nAbsolute error: %.2e\nRelative error: %.2e\n",
            gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
}