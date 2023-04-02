#include "headers/omp_ell.h"

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
void spmm_ell(ELL* mat, int threads, double* x, int k, double* y){
    int maxnz = mat->MAXNZ, rows = mat->M, *ja = mat->JA;
    double *as = mat->AS;

    int z, nz_idx;
    double *x_r, val, *t;

#pragma omp parallel for num_threads(threads) \
                                private(t, nz_idx, val, x_r, z) \
                                shared(maxnz, rows, ja, as, k, x, y) default(none)
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

//TODO:
// what if the single row has more NZs than the ones to assign to the single thread? Try to divide by blocks
// ordina le righe di ELL per numero di NZ ??
void sort_rows(ELL* ell, int *idxs){
    int idx = 0, prev_nz = 0, nz = 0, cols = ell->MAXNZ, rows = ell->M;
    Type *as = ell->AS;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (as[i*cols + j] == 0) break;
            nz++;
        }

        if (nz > prev_nz) { // put nz before prev_nz
            idxs[idx++] = nz;
            idxs[idx++] = prev_nz;
        } else {
            idxs[idx++] = prev_nz;
            idxs[idx++] = nz;
        }

        nz = 0;
    }
}

/**
 * Load balancing related to the amount of non-zeros given to each computational node.
 * The number of non-zeros is always rounded to be contained in a full row to maintain locality
 *
 * @param ts number of threads
 * */
int* ell_nz_balancing(int ts, ELL* ell, int* ordered_rows, int* rows_idx){
    int t_nz, l_ctr, g_ctr = 0, nz_count = 0;
    int rows = ell->M;
    int nnz = ell->NZ;

    //sort_rows(ell, ordered_rows);

    for (int i = 0; i < ts; i++) {
        if (i == ts-1) { // if last thread, get the remaining rows
            rows_idx[i] = rows-g_ctr;
            break;
        }

        t_nz = (((i + 1) * nnz) / ts) - ((i * nnz) / ts); // compute the number of nz to assign the i-th thread

        l_ctr = 0;
        do{
            nz_count += ordered_rows[g_ctr++];
            l_ctr++;
        }while (nz_count < t_nz && g_ctr < rows);

        rows_idx[i] = l_ctr;
    }

    return rows_idx;
}
