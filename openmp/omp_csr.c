#include "headers/omp_csr.h"

/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param rows_load balanced load of rows per thread (wrt the number of non-zeros to process)
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
//TODO: test --> avoid cache conflicts and exploit 64 bytes at a time for each row
void spmm_csr(CSR *mat, const int* rows_load, int threads, double* x, int k, double* y){

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
 * Load balancing related to the amount of non-zeros given to each computational node.
 * The number of non-zeros is always rounded to be contained in a full row to maintain locality
 *
 * OPENMP:
 *      non-zeros are balanced on the number of threads that will operate the product
 * MPI:
 *      non-zeros are balanced on the number of processes that will operate the product
 *      inside every process - openmp threads will work on the given rows in parallel
 * */
int* csr_nz_balancing(int ts, int tot_nz, const int* irp, int tot_rows){
    int i, j, r1, nz, start_row = 0, r2 = 0;

    int* rows_idx = (int*) malloc((ts+1) * sizeof(int));
    malloc_handler(1, (void*[]){rows_idx});

    for (i = 0; i < ts; i++) {
        rows_idx[i] = start_row; // add the idx of the start row

        if (i == ts-1) { // if last thread, get the remaining rows
            rows_idx[i+1] = tot_rows;
            break;
        }

        nz = ((i + 1) * tot_nz) / ts - (i * tot_nz) / ts; // compute the number of tot_nz to assign the i-th thread

        for (j = start_row; j < tot_rows; j++) {
            r2 += irp[j + 1] - irp[j]; // get number of nz in the considered rows

            if (r2 < nz) { // if the count of nz is still lower than the number of nz assigned to the thread
                r1 = r2; // save value
            } else {
                // get the number of rows that includes a number of nz closer to the one assigned
                start_row = ((r2 - nz) < (nz - r1)) ? j+1 : j;
                break;
            }
        }

        r2 = 0;
    }

    return rows_idx;
}