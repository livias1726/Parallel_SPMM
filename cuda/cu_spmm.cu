#ifdef ELLPACK
    #include "headers/cu_ell.cuh"
#else
    #include "headers/cu_csr.cuh"
#endif

int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    int k, m, n, nz;

    double flop, gflops_s, gflops_p;
    Type abs_err, rel_err;

    Type *x, *d_x, *y_s, *y_p, *d_y;

    Type *d_as;
    int *d_ja;

    StopWatchInterface* timer = 0;
    dim3 BLOCK_DIM, GRID_DIM;
    int shared_mem;

#ifdef ELLPACK
    ELL *ell;   // used for serial product and as input to compute an HELL structure
    int maxnz;
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
    flop = (double)2*k*nz;

    // ------------------------------------ Memory initialization ----------------------------------- //

    alloc_struct(&x, n, k);
    alloc_struct(&y_s, m ,k);
    alloc_struct(&y_p, m ,k);

    populate_multivector(x, n, k);

#ifdef DEBUG
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

    alloc_cuda_spmm(&d_x, &d_y, x, m, n, k);

    // convert to wanted storage format
#ifdef ELLPACK
    ell = read_mm_ell(elems, m, n, nz);
    #ifdef DEBUG
    print_ell(ell);
    #endif
#else
    csr = read_mm_csr(elems, m, n, nz);
    #ifdef DEBUG
    print_csr(csr);
    #endif
#endif

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

    // to avoid bank conflicts when double values are used
    if (!SP) checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

#ifdef ELLPACK
    maxnz = ell->MAXNZ;
    compute_ell_dimensions(m, maxnz, k, &BLOCK_DIM, &GRID_DIM, &shared_mem);
    alloc_cuda_ell(ell, &d_ja, &d_as);

    // product
    timer->start();
    spmm_ell_kernel<<<GRID_DIM, BLOCK_DIM,shared_mem>>>(m, maxnz, d_ja, d_as, d_x, k, d_y);
#else
    blocks = (int*)malloc((m+1)*sizeof(int));
    compute_csr_dimensions(csr, k, blocks, &num_blocks, &BLOCK_DIM, &GRID_DIM, &shared_mem);
    alloc_cuda_csr(csr, blocks, num_blocks, &d_irp, &d_ja, &d_as, &d_blocks);

    // product
    timer->start();
    spmm_csr_vector_kernel<<<GRID_DIM, BLOCK_DIM, shared_mem>>>(d_irp, d_ja, d_as, k, d_x, d_blocks, d_y);
#endif
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();

    gflops_p = (double)flop/((timer->getTime())*1.e6);
    checkCudaErrors(cudaMemcpy(y_p, d_y, m * k * sizeof(Type), cudaMemcpyDeviceToHost));

    // check results
    // --> double: relative error should be as close as possible to 2.22e−16 (IEEE double precision unit roundoff)
    // --> float: relative error should be as close as possible to 1.19e-07 (IEEE single precision unit roundoff)
    get_errors(m*k, y_s, y_p, &abs_err, &rel_err);

#ifdef SAVE
    save_result(y_p, m, k);
#endif

    //print_matrix(y_s, 10, k, "\nSerial Result:\n");
    //print_matrix(y_p, 10, k, "\nParallel Result:\n");

#ifdef DEBUG
    print_matrix(y_s, m, k, "\nSerial Result:\n");
    print_matrix(y_p, m, k, "\nParallel Result:\n");
#endif

    // ------------------------------------------- Cleaning up ------------------------------------------------- //
#ifdef ELLPACK
    delete[] ell;
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

    cudaDeviceReset();

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %f %f", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "Serial GFLOPS: %f\n"
                    "Parallel GFLOPS: %f\n"
                    "Absolute error: %.2e\n"
                    "Relative error: %.2e\n", gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
}