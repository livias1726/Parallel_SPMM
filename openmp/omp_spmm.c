#include "omp_utils.h"

/**
 * OpenMP version of a matrix-multivector multiplication Y <- AX where
 * - A is a MxN sparse matrix
 * - X is a Nxk dense multivector
 * */

//------------------------------------------------ SpMM ---------------------------------------------------------//
/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param rows_load balanced load of rows per thread (wrt the number of non-zeros to process)
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void product_csr(CSR *mat, const int* rows_load, int threads, double* x, int k, double* y){

    int *irp = mat->IRP, *ja = mat->JA;
    double *as = mat->AS;

    int i, j, z;
    double *t, *x_r, val;
    #pragma omp parallel for num_threads(threads) \
                                private(i, t, j, x_r, val, z) \
                                shared(threads, rows_load, irp, k, as, ja, x, y) default(none)
    for (int tid = 0; tid < threads; tid++) {
        for (i = rows_load[tid]; i < rows_load[tid + 1]; i++) { // get the specific A's row to process
            t = &y[i * k]; //the respective Y's row to accumulate products on

            for (j = irp[i]; j < irp[i+1]; j++) { // iterate over the nz values in the row
                // load just once
                x_r = &x[ja[j] * k]; //the respective X's row index
                val = as[j]; //the respective NZ value

                // Loop unrolling
                if (k % 4 == 0) { // avoids to process values like '12' with a 3-level loop unroll
                    for (z = 0; z < k; z += 4) {
                        t[z] += val * x_r[z];
                        t[z + 1] += val * x_r[z + 1];
                        t[z + 2] += val * x_r[z + 2];
                        t[z + 3] += val * x_r[z + 3];
                    }
                } else if (k % 3 == 0) {
                    for (z = 0; z < k; z += 3) {
                        t[z] += val * x_r[z];
                        t[z + 1] += val * x_r[z + 1];
                        t[z + 2] += val * x_r[z + 2];
                    }
                } else {
                    for (z = 0; z < k; z++) {
                        t[z] += val * x_r[z];
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
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
// 1. each thread gets a row from omp
// 2. in the given row, the thread reads nz value and col idx
// 3. for each nz in the row, the thread accumulates the partial products, reading x row by row
void product_ell(ELL* mat, int threads, double* x, int k, double* y){
    int maxnz = mat->MAXNZ, rows = mat->M, *ja = mat->JA;
    double *as = mat->AS;

    int z, nz_idx;
    double *x_r, val, *t;

    #pragma omp parallel for num_threads(threads) private(t, nz_idx, val, x_r, z) shared(maxnz, rows, ja, as, k, x, y) default(none)
    for (int i = 0; i < rows; i++) {
        t = &y[i*k]; // prefetching of the row to update;

        for (int j = 0; j < maxnz; j++) {
            nz_idx = i*maxnz+j;

            val = as[nz_idx];
            if (val == 0) break; // if padding is reached break loop

            x_r = &x[ja[nz_idx]*k]; // prefetching of the row in x to read from

            // Loop unrolling
            if (k % 4 == 0) { // avoids to process values like '12' with a 3-level loop unroll
                for (z = 0; z < k; z += 4) {
                    t[z] += val * x_r[z];
                    t[z+1] += val * x_r[z+1];
                    t[z+2] += val * x_r[z+2];
                    t[z+3] += val * x_r[z+3];
                }
            } else if (k % 3 == 0) {
                for (z = 0; z < k; z += 3) {
                    t[z] += val * x_r[z];
                    t[z+1] += val * x_r[z+1];
                    t[z+2] += val * x_r[z+2];
                }
            } else {
                for (z = 0; z < k; z++) {
                    t[z] += val * x_r[z];
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------------------Main
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
    product_ell(ell, num_threads, x, k, y_p);
    clock_gettime(CLOCK_MONOTONIC, &t2);

    gflops_p = GET_GFLOPS(t1, t2, flop);

    clean_up(3, (void*[]){ell->AS, ell->JA, ell});
#else

    int* rows_idx = csr_nz_balancing(num_threads, nz, csr->IRP, csr->M);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    product_csr(csr, rows_idx, num_threads, x, k, y_p);
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