#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include "../utils/utils.h"

// Simple 1-D thread block
#define BD 256

const dim3 BLOCK_DIM(BD);

// GPU implementation of matrix_vector product using a block of threads for each row.
__global__ void product_csr(int rows, int cols, const float* A, const float* x, float* y) {
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
    double *x, *y;
    int k, m, n, nz, num_threads;
    struct timespec t1, t2;
    bool ellpack;

    // check the correct use of the program
    if (argc < 4){
        fprintf(stderr, "Usage: %s [mm-filename] [storage-format] [k value]\n", argv[0]);
        exit(-1);
    }

    // create file path
#ifdef PERFORMANCE
    char path[PATH_MAX] = "resources/files/";
#else
    char path[PATH_MAX] = "../resources/files/";
#endif
    strcat(path, argv[1]);

    //check the correct opening of the matrix file
    f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        exit(-1);
    }

    // get k value and desired storage format
    ellpack = !strcmp(argv[2], "ellpack");
    k = (int)strtol(argv[3], NULL, 10);

    // process the first line of file and identify the matrix type
    if (mm_read_banner(f, &t) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(-1);
    }

    // check matrix type support
    check_mat_type(t);

    // ----------------------- Host memory initialisation ----------------------- //

    // convert to wanted storage format
    if (ellpack) {
        ell = read_mm_ell(f, t);
        m = ell->M;
        n = ell->N;
        nz = ell->NZ;
#ifdef AUDIT
        print_ell(ell);
#endif
    } else {
        csr = read_mm_csr(f, t);
        m = csr->M;
        n = csr->N;
        nz = csr->NZ;
#ifdef AUDIT
        print_csr(csr);
#endif
    }

    fclose(f);

    alloc_struct(&x, n, k);
    alloc_struct(&y, m ,k);

    populate_multivector(x, n, k);

#ifdef AUDIT
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

// ---------------------- Device memory initialisation ---------------------- //

    float *d_A, *d_x, *d_y;

    checkCudaErrors(cudaMalloc((void**) &d_A, nrows * ncols * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**) &d_x, ncols * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**) &d_y, nrows * sizeof(float)));

    // Copy matrices from the host (CPU) to the device (GPU).
    checkCudaErrors(cudaMemcpy(d_A, h_A, nrows * ncols * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x,  ncols * sizeof(float), cudaMemcpyHostToDevice));

    // ------------------------ Product on the GPU ------------------------- //

    float flopcnt=2.e-6*nrows*ncols;

    // Create the CUDA SDK timer.
    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    // DONE: Calculate the dimension of the grid of blocks (1D) needed to cover all
    // entries in the matrix and output vector
    // const dim3 GRID_DIM(ceil((nrows*ncols)/BD));
    const dim3 GRID_DIM(((nrows*ncols)+BD - 1)/BD);
    const int shmem_size = BD*sizeof(float);

    timer->start();
    gpuMatrixVector<<<GRID_DIM, BLOCK_DIM, shmem_size>>>(nrows, ncols, d_A, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    float gpuflops=flopcnt/ timer->getTime();
    std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

    // Download the resulting vector d_y from the device and store it in h_y_d.
    checkCudaErrors(cudaMemcpy(h_y_d, d_y, nrows*sizeof(float),cudaMemcpyDeviceToHost));

    // Now let's check if the results are the same.
    float reldiff = 0.0f;
    float diff = 0.0f;

    for (int row = 0; row < nrows; ++row) {
        float maxabs = std::max(std::abs(h_y[row]),std::abs(h_y_d[row]));
        if (maxabs == 0.0) {
            maxabs = 1.0;
        }
        reldiff = std::max(reldiff, std::abs(h_y[row] - h_y_d[row])/maxabs);
        diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
    }
    std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;
    // Rel diff should be as close as possible to unit roundoff;
    // float corresponds to IEEE single precision, so unit roundoff is 1.19e-07

// ------------------------------- Cleaning up ------------------------------ //

    delete timer;

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    delete[] h_A;
    delete[] h_x;
    delete[] h_y;
    delete[] h_y_d;

    return 0;
}