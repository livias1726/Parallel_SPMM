#include "headers/omp_csr.h"
#include <stdint.h>
#include <string.h>

void print512d(__m512d vec, int tid){
    double_t val[8];
    memcpy(val, &vec, sizeof(val));
    printf("%d: [%f %f %f %f %f %f %f %f]\n",
           tid, val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}

void print256i(__m256i vec, int tid){
    uint32_t val[8];
    memcpy(val, &vec, sizeof(val));
    printf("%d: [%i %i %i %i %i %i %i %i]\n",
           tid, val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}

/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param rows_load number of rows per thread
 * @param threads number of threads
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void spmm_csr_64(CSR *mat, const int* rows_load, int threads, const Type* x, int k, Type* y){

    int *irp = mat->IRP, *ja = mat->JA;
    Type *as = mat->AS;

    const __m256i scale = _mm256_set1_epi32(k), one = _mm256_set1_epi32(1);

    #pragma omp parallel for num_threads(threads) shared(threads, rows_load, irp, k, as, ja, x, y, scale, one) default(none)
    for (int tid = 0; tid < threads; tid++) {   // parallelize on threads' id

        int j, z, iter, lim;
        __m256i cols;
        __m512d vals, x_r;
        __m512d t[k];
        // init t vector
        for (z = 0; z < k; z++) { t[z] = _mm512_setzero_pd(); }

        for (int i = rows_load[tid]; i < rows_load[tid + 1]; i++) { // thread gets the row to process
            j = irp[i];
            lim = (irp[i+1] - j) / 8;

            for (iter = 0; iter < lim; iter++) {

                vals = _mm512_loadu_pd(&as[j]);                 // load 8 64-bit elements
                cols = _mm256_loadu_si256((__m256i*)&ja[j]);    // load 8 32-bit elements columns
                cols = _mm256_mullo_epi32(cols, scale);           // scale col index to be on the first column of x
                x_r = _mm512_i32gather_pd(cols, x, 8);          // build 8 64-bit elements from x and cols
                t[0] = _mm512_fmadd_pd(vals, x_r, t[0]);        // execute a fused multiply-add

                for (z = 1; z < k; z++) {
                    cols = _mm256_add_epi32(cols, one);
                    x_r = _mm512_i32gather_pd(cols, x, 8);      // build 8 64-bit elements from x and cols
                    t[z] = _mm512_fmadd_pd(vals, x_r, t[z]);    // execute a fused multiply-add
                }

                j += 8;
            }

            // remainder loop if elements are not multiple of size
            for (; j < irp[i+1]; j++) {
                for (z = 0; z < k; z++) {
                    y[i * k + z] += as[j] * x[ja[j] * k + z];
                }
            }

            // reduce all 64-bit elements in t by addition
            for (z = 0; z < k; z++) {
                y[i * k + z] += _mm512_reduce_add_pd(t[z]);
                t[z] = _mm512_setzero_pd();
            }
        }
    }
}

void spmm_csr(CSR *mat, const int* rows_load, int threads, const Type* x, int k, Type* y){
    if (sizeof(Type) == 8) {
        spmm_csr_64(mat, rows_load, threads, x, k, y);
    } else {
        //spmm_csr_32();
    }
}
 /*
void spmm_csr(CSR *mat, const int* rows_load, int threads, Type* x, int k, Type* y){

    int *irp = mat->IRP, *ja = mat->JA;
    Type *as = mat->AS;

#pragma omp parallel for num_threads(threads) shared(threads, rows_load, irp, k, as, ja, x, y) default(none)
    for (int tid = 0; tid < threads; tid++) {
        int j, z;
        Type *t, *x_r, val;

        for (int i = rows_load[tid]; i < rows_load[tid + 1]; i++) { // get the specific A's row to process
            t = &y[i * k]; //the respective Y's row to accumulate products on

            for (j = irp[i]; j < irp[i+1]; j++) { // iterate over the nz values in the row
                // load just once
                x_r = &x[ja[j] * k];    //the respective X's row index
                val = as[j];            //the respective NZ value

                // loop unrolling
                if (k % 4 == 0) {
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
  */


/**
 * Load balancing related to the amount of non-zeros given to each thread.
 * The number of non-zeros is always rounded to be contained in a full row to maintain locality.
 * */
void csr_nz_balancing(int threads, int tot_nz, const int* irp, int tot_rows, int* rows_idx){
    int j, nz, nz_prev = 0, nz_curr, start_row = 0, r_prev, r_curr = 0;

    for (int i = 0; i < threads; i++) {
        rows_idx[i] = start_row; // add the idx of the start row

        // compute the number of non-zeros to assign the i-th thread
        nz_curr = ((i + 1) * tot_nz) / threads;
        nz = nz_curr - nz_prev;
        nz_prev = nz_curr;

        for (j = start_row; j < tot_rows; j++) {
            r_curr += irp[j + 1] - irp[j]; // get number of nz in the considered rows

            if (r_curr < nz) { // if the count of nz is still lower than the number of nz assigned to the thread
                r_prev = r_curr; // save value
            } else {
                // get the number of rows that includes a number of nz closer to the one assigned
                start_row = ((r_curr - nz) < (nz - r_prev)) ? j + 1 : j;
                break;
            }
        }

        r_curr = 0;
    }

    rows_idx[threads] = tot_rows; // last thread gets the remaining rows
}