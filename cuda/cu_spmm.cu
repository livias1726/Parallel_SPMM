#include "cu_utils.cuh"

/*
__device__ void product_csr_stream(int* irp, int* ja, double* as, int start, int end, int idx, const double* x,
                                   double* y){
    int fc = irp[start];
    int nz = irp[end] - fc;

    __shared__ volatile double LDS[BD];

    // each thread writes to shared memory
    if (idx < nz){
        LDS[idx] = as[fc + idx] * x[ja[fc + idx]];
    }
    __syncthreads();

    // threads that fall within a range sum up the partial results
    int k, j;
    for (k = start + idx; k < end; k += blockDim.x){
        double temp = 0.0;

        for (j = (irp[k] - fc); j < (irp[k + 1] - fc); j++){
            temp = temp + LDS[j];
        }
        y[k] = temp;
    }
}
 */

 /**
  * VECTOR KERNEL: 1 warp per matrix row
  *     - A warp-wide parallel reduction is required to sum the per-thread results together
  *     - Accesses JA and AS contiguously
  *     - Memory access labeled by warp index
  *     - If rows don't have more than 32 NZs each, then no warp iterates more than once on the CSR arrays.
  *       Else, the order of summation differs from the scalar kernel
  * Problems: efficient execution demands a number of NZs per row greater than 32 (warpSize)
  * */
 /*
__device__ void product_csr_vector(int* irp, int* ja, double* as, int start, int end, int idx, const double* x,
                                   double* y){
    // tid in warp
    int rs = irp[start];
    int re = irp[end];

    double sum = 0.0;

    __shared__ volatile double LDS[BD];

    // use all threads in a warp to accumulate multiplied elements
    for (int j = rs + idx; j < re; j += BD){
        int col = ja[j];
        sum += as[j] * x[col];
    }

    LDS[idx] = sum;
    __syncthreads();

    // Reduce partial sums
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (idx < stride) LDS[idx] += LDS[idx + stride];
    }

    // write result
    if (idx == 0) y[start] = LDS[idx];
}
  */

/*
__global__ void product_csr_adaptive(CSR* csr, int* blocks, const double* x, double* y) {
    int start = blocks[blockIdx.x];
    int end = blocks[blockIdx.x + 1];

    int rows = end - start;
    int idx = threadIdx.x;

    if (rows > 1) {
        product_csr_stream(csr->IRP, csr->JA, csr->AS, start, end, idx, x, y);
    } else {
        product_csr_vector(csr->IRP, csr->JA, csr->AS, start, end, idx, x, y);
    }
}
 */

/**
 * SCALAR KERNEL: 1 thread per matrix row
 * Problems: Thread divergence
 *      - JA and AS not accessed simultaneously
 *      - If the distribution of NZs per row is highly variable (follows a power law),
 *        many threads within a warp will remain idle while waiting for the thread with the longest row.
 * */
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
    /* SCALAR */
    dim3 GRID_DIM;
    get_grid(m, GRID_DIM);

    timer->start();
    product_csr_scalar<<<GRID_DIM, BD>>>(m, d_irp, d_ja, d_as, d_x, k, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();

    // compute row blocks
    /*
    int *d_blocks, *blocks = (int*)malloc(m*sizeof(int));
    int num_blocks = csr_adaptive_blocks(m, csr->IRP, blocks);
    checkCudaErrors(cudaMalloc((void**) &d_blocks, num_blocks*sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_blocks, blocks, num_blocks*sizeof(int), cudaMemcpyHostToDevice));
    product_csr_adaptive<<<num_blocks-1, BD>>>(d_csr, d_blocks, d_x, d_y);
     */
#endif

    gflops_p = (double)flop/((timer->getTime())*1.e6);
    checkCudaErrors(cudaMemcpy(y_p, d_y, m * k * sizeof(double), cudaMemcpyDeviceToHost));

    // check results
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