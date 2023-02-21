#include "omp_utils.h"

/**
 * omp_spmm performs an OpenMP multithreaded version of a matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 * */

//---------------------------------------------------------------------------------------------------Pre-processing
void process_arguments(int argc, char** argv, FILE *f, bool* ell_flag, int* k){
    if (argc < 5){
        fprintf(stderr, "Usage: %s [mm-filename] [storage-format] [k value] [num-threads]\n", argv[0]);
        exit(-1);
    }

    // create file path
    char path[PATH_MAX] = "../resources/files/";
    strcat(path, argv[1]);

    //check the correct opening of the matrix file
    f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        exit(-1);
    }

    // get k value and desired storage format
    *ell_flag = !strcmp(argv[2], "ellpack");
    *k = (int)strtol(argv[3], NULL, 10);
}

//---------------------------------------------------------------------------------------------------Product
/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param nz_start array of starting row indices for each thread
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * @param t1 pointer to first timeval structure
 * @param t2 pointer to second timeval structure
 * */
void product_csr(CSR mat, const int* nz_start, int num_threads, const double* x, int k, double* y,
                 struct timespec *t1, struct timespec *t2){
    int i, j, z, lim_col, lim_row, rows = mat.M;
    double temp;

    clock_gettime(CLOCK_MONOTONIC, t1);

    #pragma omp parallel for private(i, j, z, lim_col, lim_row, temp) shared(nz_start,num_threads,k,y,mat,rows, x) default(none)
    for (int t = 0; t < num_threads; t++) {

        lim_row = (t != num_threads-1) ? nz_start[t+1] : rows;
        for (i = nz_start[t]; i < lim_row; i++) {

            lim_col = (i != rows-1) ? mat.IRP[i+1] : mat.NZ;
            for (z = 0; z < k; z++) {
                temp = 0.0;

                for (j = mat.IRP[i]; j < lim_col; j++) {
                    temp += mat.AS[j]*(x[mat.JA[j]*k+z]);
                }

                y[i*k+z] = temp;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, t2);
}

/**
 * Computes the product with A stored in a ELL format
 *
 * @param mat matrix in ellpack format
 * @param x multivector Nxk stored as 1D array
 * @param y receives product results Mxk stored as 1D array
 * @param t1 pointer to first timeval structure
 * @param t2 pointer to second timeval structure
 * */
void product_ell(ELL mat, const double* x, int k, double* y, struct timespec *t1, struct timespec *t2){
    int i, j, z, maxnz = mat.MAXNZ;
    double t, val;

    clock_gettime(CLOCK_MONOTONIC, t1);
    // TODO: version 1 -> to be optimized
    #pragma omp parallel for schedule(guided) shared(k, maxnz, x, mat, y) private(z, t, j, val) default(none)
    for (i = 0; i < mat.M; i++) {
        for (z = 0; z < k; z++) { //TODO: check order of loops
            t = 0.0;

            for (j = 0; j < maxnz; j++) {
                val = mat.AS[i*maxnz+j];
                if (val == 0) { // if padding is reached break loop
                    break;
                }
                t += val*x[mat.JA[i*maxnz+j]*k+z];
            }

            y[i*k+z] = t;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, t2);
}

//-----------------------------------------------------------------------------------------Main
int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR* csr;
    ELL* ell;
    long time;
    double gflops_s, gflops_p, abs_err, rel_err;
    double *x, *y_s, *y_p;
    int k, m, n, nz, num_threads;
    struct timespec t1, t2;
    bool ellpack;

    process_arguments(argc, argv, f, &ellpack, &k);
    process_mm(&t, f);

    // set number of threads
    num_threads = (int)strtol(argv[4], NULL, 10);
    omp_set_num_threads(num_threads);

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
    alloc_struct(&y_s, m ,k);
    alloc_struct(&y_p, m ,k);

    populate_multivector(x, n, k);

#ifdef AUDIT
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

    // compute the product
    if (ellpack) {
        serial_product_ell(*ell, x, k, y_s, &t1, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_s = get_gflops(time, k, nz);

        //TODO: optimize ell prod
        product_ell(*ell, x, k, y_p, &t1, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_p = get_gflops(time, k, nz);

        free(ell);
    } else {
        serial_product_csr(*csr, x, k, y_s, &t1, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_s = get_gflops(time, k, nz);

        int* nz_start = nz_balancing(num_threads, nz, csr->IRP, csr->M);
        product_csr(*csr, nz_start, num_threads, x, k, y_p, &t1, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_p = get_gflops(time, k, nz);

        free(nz_start);
        free(csr);
    }

    abs_err = get_absolute_error(m*k, y_s, y_p);
    rel_err = get_relative_error(m*k, abs_err, y_s);

    //save_result(y, m, k);
    free(y_s);
    free(y_p);
    free(x);

#ifdef AUDIT
    // print results
    print_matrix(y, m, k, "\nResult:\n");
#endif

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %f %f", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "\nSerial GFLOPS: %f\nParallel GFLOPS: %f\nAbsolute error: %f\nRelative error: %f\n",
            gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
} 