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
                                   const double *x, const int k, double *y){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < rows){
        double temp = 0.0;
        int row_start = irp[row];
        int row_end = irp[row+1];

        for(int z = 0; z < k; z++){
            for (int j = row_start; j < row_end; j ++)
                temp += as[j] * x[ja[j]*k + z];
        }

        y[row] += temp;
    }
}

int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR *csr;
    //ELL *ell, *d_ell;
    double gflops_s, gflops_p, abs_err, rel_err;
    double *x, *y_s, *y_p, *d_x, *d_y;
    int k, m, n, nz, flop;
    int *d_irp, *d_ja;
    double *d_as;

    // ----------------------- Set Up ------------------------------------------- //

    process_arguments(argc, argv, &f, &k);
    process_mm(&t, f);

    // ----------------------- Host memory initialisation ----------------------- //

    // convert to wanted storage format
#ifdef ELLPACK
    ell = read_mm_ell(f, t);
    m = ell->M;
    n = ell->N;
    nz = ell->NZ;

    #ifdef DEBUG
    print_ell(ell);
    #endif

    checkCudaErrors(cudaMalloc((void**) &d_ell, sizeof(ELL)));
    checkCudaErrors(cudaMemcpy(d_ell, ell, sizeof(ELL), cudaMemcpyHostToDevice));
#else
    csr = read_mm_csr(f, t);
    m = csr->M;
    n = csr->N;
    nz = csr->NZ;

    #ifdef DEBUG
    print_csr(csr);
    #endif

    allocCudaCsr(csr, &d_irp, &d_ja, &d_as);
#endif

    fclose(f);

    flop = 2*k*nz;

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

    // ------------------------ Product on the CPU ------------------------- //
    timer->start();
#ifdef ELLPACK
    serial_product_ell(ell, x, k, y_s);
#else
    serial_product_csr(csr, x, k, y_s);
#endif
    timer->stop();

    gflops_s = (double)flop/((timer->getTime())*1.e6);

    // ------------------------ Product on the GPU ------------------------- //
#ifdef ELLPACK
    //TODO
#else
    // compute row blocks
    int *d_blocks, *blocks = (int*)malloc(m*sizeof(int));
    int num_blocks = csr_adaptive_blocks(m, csr->IRP, blocks);
    checkCudaErrors(cudaMalloc((void**) &d_blocks, num_blocks*sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_blocks, blocks, num_blocks*sizeof(int), cudaMemcpyHostToDevice));

    timer->start();
    product_csr_scalar<<<num_blocks-1,BD>>>(m, d_irp, d_ja, d_as, d_x, k, d_y);
    //product_csr_adaptive<<<num_blocks-1, BD>>>(d_csr, d_blocks, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
#endif

    gflops_p = (double)flop/((timer->getTime())*1.e6);
    checkCudaErrors(cudaMemcpy(y_p, d_y, m * k * sizeof(double), cudaMemcpyDeviceToHost));

    // check results
    abs_err = get_absolute_error(m*k, y_s, y_p);
    rel_err = get_relative_error(m*k, abs_err, y_s);

// ------------------------------- Cleaning up ------------------------------ //
    delete timer;

#ifdef ELLPACK
    checkCudaErrors(cudaFree(d_ell));
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

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %f %f", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "\nSerial GFLOPS: %f\nParallel GFLOPS: %f\nAbsolute error: %f\nRelative error: %f\n",
            gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
}