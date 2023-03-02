#include "omp_utils.h"

/**
 * OpenMP version of a matrix-multivector multiplication Y <- AX where
 * - A is a MxN sparse matrix
 * - X is a Nxk dense multivector
 * */

//---------------------------------------------------------------------------------------------------Product
/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param rows_load balanced load of rows per thread (wrt the number of non-zeros to process)
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
//TODO: try to parallelize the iteration on x --> each thread has a block of y (rows_load x cols_load)
// --> cols_load will be the columns of x that the thread has to manage
void product_csr(CSR *mat, const int* rows_load, int threads, double* x, int k, double* y){

    int *irp = mat->IRP, *ja = mat->JA;
    double *as = mat->AS;

    int z;
#pragma omp parallel for num_threads(threads) private(z) shared(threads, rows_load, irp, k, as, ja, x, y) default(none)
    for (int tid = 0; tid < threads; tid++) {
        int i, j;
        double *row_tmp, *col_tmp, a_j;

        for (i = rows_load[tid]; i < rows_load[tid + 1]; i++) { // get the specific A's row to process
            row_tmp = &y[i * k]; //the respective Y's row to accumulate products on

            for (j = irp[i]; j < irp[i+1]; j++) { // iterate over the nz values in the row
                // load just once
                col_tmp = &x[ja[j]*k]; //the respective X's row index
                a_j = as[j]; //the respective NZ value

                // Loop unrolling
                if (k % 4 == 0) { // avoids to process values like '12' with a 3-level loop unroll
                    for (z = 0; z < k; z += 4) {
                        row_tmp[z] += a_j * col_tmp[z];
                        row_tmp[z+1] += a_j * col_tmp[z+1];
                        row_tmp[z+2] += a_j * col_tmp[z+2];
                        row_tmp[z+3] += a_j * col_tmp[z+3];
                    }
                } else if (k % 3 == 0) {
                    for (z = 0; z < k; z += 3) {
                        row_tmp[z] += a_j * col_tmp[z];
                        row_tmp[z+1] += a_j * col_tmp[z+1];
                        row_tmp[z+2] += a_j * col_tmp[z+2];
                    }
                } else {
                    for (z = 0; z < k; z++) {
                        row_tmp[z] += a_j * col_tmp[z];
                    }
                }
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
    double gflops_s, gflops_p, abs_err, rel_err, *x, *y_s, *y_p;
    int k, m, n, nz, num_threads;
    struct timespec t1, t2;

    process_arguments(argc, argv, &f, &k, &num_threads);
    process_mm(&t, f);

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

    double flop = (double)2*k*nz;

    // ----------------------------------- Serial SpMM -------------------------------------------- //

    clock_gettime(CLOCK_MONOTONIC, &t1);
#ifdef ELLPACK
    serial_product_ell(*ell, x, k, y_s);
#else
    serial_product_csr(csr, x, k, y_s);
#endif
    clock_gettime(CLOCK_MONOTONIC, &t2);
    gflops_s = GET_GFLOPS(t1, t2, flop);

    // -------------------------------------------- OpenMP SpMM ---------------------------------------------- //
#ifdef ELLPACK
    clock_gettime(CLOCK_MONOTONIC, &t1);

    //TODO
    product_ell(*ell, x, k, y_p, &t1, &t2);

    clock_gettime(CLOCK_MONOTONIC, &t2);
    gflops_p = GET_GFLOPS(t1, t2, flop);

    free(ell);
#else
    int* rows_idx = nz_balancing(num_threads, nz, csr->IRP, csr->M);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    product_csr(csr, rows_idx, num_threads, x, k, y_p);

    clock_gettime(CLOCK_MONOTONIC, &t2);

    gflops_p = GET_GFLOPS(t1, t2, flop);

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