#ifdef ELLPACK
    #include "headers/cu_hll.cuh"
#else
    #include "headers/cu_csr.cuh"
#endif

int main(int argc, char** argv) {

    // matrices
    MM_typecode t;
    FILE *f;
    int k, m, n, nz, z;
    Type *x, *d_x, *y_s, *y_p, *d_y;
    Type *d_as;
    int *d_ja;

    // performance
    float flop, gflops_s, gflops_p;
    Type abs_err, rel_err;
    StopWatchInterface *timer = 0;

    // cuda dimensioning
    dim3 BLOCK_DIM, GRID_DIM;
    int shared_mem;

#ifdef ELLPACK
    ELL *ell;   // used for serial product and as input to compute an HELL structure
    HLL *hll; // used for gpu product
    int *d_maxnz, *d_hack_offset;
#else
    CSR *csr;
    int num_blocks, *blocks;
    int *d_irp, *d_blocks;
#endif

    // -------------------------------------- Set Up ------------------------------------------- //

    // parse command line and input matrix
    process_arguments(argc, argv, &f, &k);
    process_mm(&t, f);

    // read matrix from file
    Elem** elems = read_mm(f, &m, &n, &nz, t);
    fclose(f);

    // timer
    sdkCreateTimer(&timer);

    // flops
    flop = (float)2*k*nz;

    // ------------------------------------ Memory initialization ----------------------------------- //

    alloc_struct(&x, n, k);
    alloc_struct(&y_s, m ,k);
    alloc_struct(&y_p, m ,k);

    populate_multivector(x, n, k);
    //print_matrix(x, n, k, "\nMultivector:\n");

    // convert to wanted storage format
#ifdef ELLPACK
    ell = read_mm_ell(elems, m, n, nz);
    //print_ell(ell);
#else
    csr = read_mm_csr(elems, m, n, nz);
    //print_csr(csr);
#endif

    // ------------------------------------------- Serial CPU SpMM --------------------------------------------- //
    gflops_s = 0;
    for (z = 0; z < MAX_NUM_RUNS; ++z) {
        timer->start();
#ifdef ELLPACK
        serial_product_ell(ell, x, k, y_s);
#else
        serial_product_csr(csr, x, k, y_s);
#endif
        timer->stop();

        gflops_s += (float)flop/((timer->getTime())*1.e6);
        timer->reset();
    }

    gflops_s /= MAX_NUM_RUNS;

    // --------------------------------------------- GPU SpMM -------------------------------------------------- //

    // dimensioning and allocation
#ifdef ELLPACK
    compute_hll_dimensions(ell, k, &hll, &BLOCK_DIM, &GRID_DIM, &shared_mem);
    //print_hll(hll, GRID_DIM.x);
    alloc_cuda_hll(hll, GRID_DIM.x, &d_maxnz, &d_hack_offset, &d_ja, &d_as);
#else
    blocks = (int*)malloc((m+1)*sizeof(int));
    compute_csr_dimensions(csr, k, blocks, &num_blocks, &BLOCK_DIM, &GRID_DIM, &shared_mem);
    alloc_cuda_csr(csr, blocks, num_blocks, &d_irp, &d_ja, &d_as, &d_blocks);
#endif

    // to avoid bank conflicts when double values are used
    if (sizeof(Type) == 8) checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    alloc_cuda_spmm(&d_x, &d_y, x, m, n, k);

    // product
    gflops_p = 0;
    for (z = 0; z < MAX_NUM_RUNS; ++z) {
        timer->start();
#ifdef ELLPACK
        spmm_hll_kernel<<<GRID_DIM, BLOCK_DIM,shared_mem>>>(m, d_maxnz, d_hack_offset, d_ja, d_as, d_x, k, d_y);
#else
        spmm_csr_vector_kernel<<<GRID_DIM, BLOCK_DIM, shared_mem>>>(d_irp, d_ja, d_as, k, d_x, d_blocks, d_y);
#endif
        checkCudaErrors(cudaDeviceSynchronize());
        timer->stop();

        gflops_p += (float)flop/((timer->getTime())*1.e6);
        timer->reset();
        //if (z < MAX_NUM_RUNS-1) checkCudaErrors(cudaMemset(d_y, 0, m * k * sizeof(Type)));
    }

    gflops_p /= MAX_NUM_RUNS;

    checkCudaErrors(cudaMemcpy(y_p, d_y, m * k * sizeof(Type), cudaMemcpyDeviceToHost));

    // -------------------------------------------- check errors ---------------------------------------------------- //
    get_errors(m*k, y_s, y_p, &abs_err, &rel_err);

#ifdef SAVE
    save_result(argv[1], y_p, m, k);
#endif

    //print_matrix(y_s, m, k, "\nSerial Result:\n");
    //print_matrix(y_p, m, k, "\nParallel Result:\n");

    // ------------------------------------------- Clean up ------------------------------------------------- //
#ifdef ELLPACK
    checkCudaErrors(cudaFree(d_maxnz));
    checkCudaErrors(cudaFree(d_hack_offset));
    delete[] hll;
#else
    checkCudaErrors(cudaFree(d_irp));
    checkCudaErrors(cudaFree(d_blocks));
    delete[] csr;
#endif

    delete timer;

    checkCudaErrors(cudaFree(d_ja));
    checkCudaErrors(cudaFree(d_as));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    delete[] x;
    delete[] y_s;
    delete[] y_p;

    checkCudaErrors(cudaDeviceReset());

    // ---------------------------------------------- Results -------------------------------------------------- //
#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %.2e %.2e", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "Serial GFLOPS: %f\n"
                    "Parallel GFLOPS: %f\n"
                    "Absolute error: %.2e\n"
                    "Relative error: %.2e\n", gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
}