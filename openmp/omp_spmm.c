#include "omp_utils.h"

/**
 * omp_spmm performs an OpenMP multithreaded version of a matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 * */

//---------------------------------------------------------------------------------------------------Product
/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param nz_start array of starting row indices for each thread
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void product_csr(CSR mat, const int* nz_start, int num_threads, const double* x, int k, double* y){
    int rows = mat.M;

    #pragma omp parallel for num_threads(num_threads) shared(nz_start,num_threads,k,y,mat,rows, x) default(none)
    for (int tid = 0; tid < num_threads; tid++) {
        int limit, z, j;
        double temp;

        for (int i = nz_start[tid]; i < nz_start[tid+1]; i++) {
            limit = (i != rows-1) ? mat.IRP[i+1] : mat.NZ;

            for (z = 0; z < k; z++) {
                temp = 0.0;

                for (j = mat.IRP[i]; j < limit; j++) {
                    temp += mat.AS[j]*(x[mat.JA[j]*k+z]);
                }

                y[i*k+z] = temp;
            }
        }
    }
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
    double gflops_s, gflops_p, abs_err, rel_err, *x, *y_s, *y_p;
    int k, m, n, nz, num_threads;
    struct timespec t1, t2;

    process_arguments(argc, argv, &f, &k, &num_threads);
    process_mm(&t, f);

    // convert to wanted storage format
#ifdef ELLPACK
    ell = read_mm_ell(f, t);
    m = ell->M;
    n = ell->N;
    nz = ell->NZ;
    #ifdef DEBUG
    print_ell(ell);
    #endif
#else
    csr = read_mm_csr(f, t);
    m = csr->M;
    n = csr->N;
    nz = csr->NZ;
    #ifdef DEBUG
    print_csr(csr);
    #endif
#endif
    fclose(f);

    alloc_struct(&x, n, k);
    alloc_struct(&y_s, m ,k);
    alloc_struct(&y_p, m ,k);

    populate_multivector(x, n, k);

#ifdef DEBUG
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

    // compute the product
#ifdef ELLPACK
    serial_product_ell(*ell, x, k, y_s, &t1, &t2);
    time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
    gflops_s = get_gflops(time, k, nz);

    //TODO: optimize ell prod
    product_ell(*ell, x, k, y_p, &t1, &t2);
    time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
    gflops_p = get_gflops(time, k, nz);

    free(ell);
#else
    clock_gettime(CLOCK_MONOTONIC, &t1);
    serial_product_csr(*csr, x, k, y_s);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
    gflops_s = get_gflops(time, k, nz);

    int* rows_idx = nz_balancing(num_threads, nz, csr->IRP, csr->M);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    product_csr(*csr, rows_idx, num_threads, x, k, y_p);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
    gflops_p = get_gflops(time, k, nz);

    free(rows_idx);
    free(csr);
#endif

    abs_err = get_absolute_error(m*k, y_s, y_p);
    rel_err = get_relative_error(m*k, abs_err, y_s);
#ifdef SAVE
    save_result(y_p, m, k);
#endif

    free(x);

#ifdef DEBUG
    // print results
    print_matrix(y_s, m, k, "\nSerial Result:\n");
    print_matrix(y_p, m, k, "\nParallel Result:\n");
#endif

    free(y_s);
    free(y_p);

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f %f %f", gflops_s, gflops_p, abs_err, rel_err);
#else
    fprintf(stdout, "Serial GFLOPS: %f\nParallel GFLOPS: %f\nAbsolute error: %f\nRelative error: %f\n",
            gflops_s, gflops_p, abs_err, rel_err);
#endif

    return 0;
} 