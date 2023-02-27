#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include "cu_utils.h"

// Simple 1-D thread block
#define BD 256

const dim3 BLOCK_DIM(BD);

// GPU implementation of matrix_vector product using a block of threads for each row.
__global__ void gpuProductCsr(int rows, int cols, const float* A, const float* x, float* y) {
    // use of shared memory
    extern __shared__ float shared_row[];

    unsigned int row = blockIdx.x;
    if (row < rows) { // which row is the current block supposed to act upon?

        // each thread processes the elements at indexes with period tid
        unsigned int tid = threadIdx.x;
        shared_row[tid] = 0.0;
        for (unsigned int i = tid; i < cols; i += blockDim.x) {
            shared_row[tid] += A[row*cols + i] * x[i];
        }
        __syncthreads();

        // implementation of a reduction operation (V3)
        for(unsigned int s=blockDim.x>>1; s>0; s>>=1) {
            if (tid < s) {
                shared_row[tid] += shared_row[tid + s];
            }
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            y[row] = shared_row[0];
        }
    }
}

int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR* csr;
    ELL* ell;
    double gflops_s, gflops_p, abs_err, rel_err;
    double *x, *y_s, *y_p;
    int k, m, n, nz;

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

    ELL *d_ell;
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

    CSR *d_csr;
    checkCudaErrors(cudaMalloc((void**) &d_csr, sizeof(CSR)));
    checkCudaErrors(cudaMemcpy(d_csr, csr, sizeof(CSR), cudaMemcpyHostToDevice));
#endif

    fclose(f);

    alloc_struct(&x, n, k);
    alloc_struct(&y_s, m ,k);
    alloc_struct(&y_p, m ,k);

    populate_multivector(x, n, k);

#ifdef AUDIT
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

    double *d_x, *d_y;
    checkCudaErrors(cudaMalloc((void**) &d_x, n * k * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_y, m * k * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_x, x,  n * k * sizeof(double), cudaMemcpyHostToDevice));


    // CUDA SDK timer
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

    gflops_s = get_gflops((timer->getTime())*1.e6, k, nz);

    // ------------------------ Product on the GPU ------------------------- //
    // TODO: compute the dimension of the grid of blocks (1D) needed to cover all entries in the matrix and output vector
    const dim3 GRID_DIM(((nrows*ncols)+BD - 1)/BD);
    const int shmem_size = BD*sizeof(float);

    timer->start();
    gpuProductCsr<<<GRID_DIM, BLOCK_DIM, shmem_size>>>(nrows, ncols, d_A, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();

    gflops_p = get_gflops((timer->getTime())*1.e6, k, nz);

    // get the resulting vector d_y from the device and store it in y
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
    checkCudaErrors(cudaFree(d_csr));
    delete[] csr;
#endif

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