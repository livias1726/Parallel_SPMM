#ifdef ELLPACK
    #include "headers/omp_ell.h"
#else
    #include "headers/omp_csr.h"
#endif

/**
 * OpenMP version of a matrix-multivector multiplication Y <- AX where
 * - A is a MxN sparse matrix
 * - X is a Nxk dense multivector
 * */

int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR* csr;
    ELL* ell;
    double flop, gflops_s, gflops_p, abs_err, rel_err, *x, *y_s, *y_p;
    int k, m, n, nz, num_threads;
    struct timespec t1, t2;

    process_arguments(argc, argv, &f, &k, &num_threads);
    process_mm(&t, f);

    // ------------------------------------------------ Pre-processing ------------------------------------------- //
    // read matrix from file
    Elem** elems = read_mm(f, &m, &n, &nz, t);
    fclose(f);

    // convert to wanted storage format
#ifdef ELLPACK
    //TODO: manage H-Ellpack
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

    alloc_struct(&x, n, k);
    alloc_struct(&y_s, m ,k);
    alloc_struct(&y_p, m ,k);

    populate_multivector(x, n, k);

#ifdef DEBUG
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

    flop = (double)2*k*nz;

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
    /*int* ordered_rows_idx = (int*) malloc(m * sizeof(int));
    int* rows_displ = (int*) malloc((num_threads+1) * sizeof(int));
    malloc_handler(2, (void*[]){ordered_rows_idx, rows_displ});

    ell_nz_balancing(num_threads, ell, ordered_rows_idx, rows_displ);*/

    clock_gettime(CLOCK_MONOTONIC, &t1);
    spmm_ell(ell, num_threads, x, k, y_p);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    gflops_p = GET_GFLOPS(t1, t2, flop);

    clean_up(3, (void*[]){ell->AS, ell->JA, ell});
#else

    int* rows_idx = csr_nz_balancing(num_threads, nz, csr->IRP, csr->M);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    spmm_csr(csr, rows_idx, num_threads, x, k, y_p);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    gflops_p = GET_GFLOPS(t1, t2, flop);

    clean_up(5, (void*[]){csr->AS, csr->JA, csr->IRP, csr, rows_idx});
#endif

    // check results
    get_errors(m, k, y_s, y_p, &abs_err, &rel_err);
#ifdef SAVE
    save_result(y_p, m, k);
#endif
#ifdef DEBUG
    print_matrix(y_s, m, k, "\nSerial Result:\n");
    print_matrix(y_p, m, k, "\nParallel Result:\n");
#endif

    clean_up(3, (void*[]){x, y_s, y_p});

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %f %f", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "Serial GFLOPS: %f\nParallel GFLOPS: %f\nAbsolute error: %.2e\nRelative error: %.2e\n",
            gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
}