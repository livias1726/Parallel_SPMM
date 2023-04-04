#ifdef ELLPACK
    #include "headers/omp_ell.h"
#else
    #include "headers/omp_csr.h"
#endif

int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    int k, m, n, nz, num_threads;
    double flop, gflops_s, gflops_p;
    Type abs_err, rel_err;
    Type *x, *y_s, *y_p;
    struct timespec t1, t2;

#ifdef ELLPACK
    ELL *ell;
#else
    CSR *csr;
    int* thread_rows;
#endif

    // ------------------------------------------------ Set Up ------------------------------------------- //

    // parse command line and input matrix
    process_arguments(argc, argv, &f, &k, &num_threads);
    process_mm(&t, f);

    // read matrix from file
    Elem** elems = read_mm(f, &m, &n, &nz, t);
    fclose(f);

    // flops
    flop = (double)2*k*nz;

    // ------------------------------------ Memory initialization ----------------------------------- //

    alloc_struct(&x, n, k);
    alloc_struct(&y_s, m ,k);
    alloc_struct(&y_p, m ,k);

    populate_multivector(x, n, k);
#ifdef DEBUG
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

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

    // ----------------------------------------------- Serial SpMM -------------------------------------------- //

    clock_gettime(CLOCK_MONOTONIC, &t1);
#ifdef ELLPACK
    serial_product_ell(ell, x, k, y_s);
#else
    serial_product_csr(csr, x, k, y_s);
#endif
    clock_gettime(CLOCK_MONOTONIC, &t2);
    gflops_s = GET_GFLOPS(t1, t2, flop);

    // ----------------------------------------------- OpenMP SpMM ---------------------------------------------- //

#ifdef ELLPACK
    for (int s = 0; s < num_threads; s++) printf("."); // TODO: why this print enormously speed omp product???
    printf("\n");

    clock_gettime(CLOCK_MONOTONIC, &t1);
    spmm_ell(ell, num_threads, x, k, y_p);
#else
    thread_rows = (int*) malloc((num_threads + 1) * sizeof(int));
    malloc_handler(1, (void*[]){thread_rows});
    csr_nz_balancing(num_threads, nz, csr->IRP, csr->M, thread_rows);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    spmm_csr(csr, thread_rows, num_threads, x, k, y_p);
#endif
    clock_gettime(CLOCK_MONOTONIC, &t2);

    gflops_p = GET_GFLOPS(t1, t2, flop);

    // check results
    // --> double: relative error should be as close as possible to 2.22e−16 (IEEE double precision unit roundoff)
    // --> float: relative error should be as close as possible to 1.19e-07 (IEEE single precision unit roundoff)
    get_errors(m*k, y_s, y_p, &abs_err, &rel_err);

#ifdef SAVE
    save_result(argv[1], y_p, m, k);
#endif

#ifdef DEBUG
    print_matrix(y_s, m, k, "\nSerial Result:\n");
    print_matrix(y_p, m, k, "\nParallel Result:\n");
#endif

    // ------------------------------------------- Clean up ------------------------------------------------- //
#ifdef ELLPACK
    clean_up(3, (void*[]){ell->AS, ell->JA, ell});
#else
    clean_up(5, (void*[]){csr->AS, csr->JA, csr->IRP, csr, thread_rows});
#endif

    clean_up(3, (void*[]){x, y_s, y_p});

    // ---------------------------------------------- Results -------------------------------------------------- //
#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %lf %lf", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "Serial GFLOPS: %f\n"
                    "Parallel GFLOPS: %f\n"
                    "Absolute error: %.2e\n"
                    "Relative error: %.2e\n", gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
}