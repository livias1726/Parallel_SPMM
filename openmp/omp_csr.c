#include "headers/omp_csr.h"
#include <stdint.h>
#include <string.h>

void print512d(__m512d vec, int tid, int dim){
    double_t val[dim];
    memcpy(val, &vec, sizeof(val));

    printf("%d: [", tid);
    for (int i = 0; i < dim; i++) {
        printf("%f ", val[i]);
    }
    printf("]\n");
}

void print512(__m512 vec, int tid, int dim){
    float_t val[dim];
    memcpy(val, &vec, sizeof(val));

    printf("%d: [", tid);
    for (int i = 0; i < dim; i++) {
        printf("%f ", val[i]);
    }
    printf("]\n");
}

void print512i(__m512i vec, int tid, int dim){
    uint32_t val[dim];
    memcpy(val, &vec, sizeof(val));

    printf("%d: [", tid);
    for (int i = 0; i < dim; i++) {
        printf("%i ", val[i]);
    }
    printf("]\n");
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

    int *irp = mat->IRP;
    int *ja = mat->JA;
    Type *as = mat->AS;

    const __m256i scale = _mm256_set1_epi32(k);

    #pragma omp parallel for num_threads(threads) shared(threads, rows_load, irp, k, as, ja, x, y, scale) default(none)
    for (int tid = 0; tid < threads; tid++) {   // parallelize on threads' id
        // private params
        int j, z, iter, lim, r_y, r_x;
        Type val;
        __m256i cols;
        __m512d vals, x_r;

        __m512d t[k];
        #pragma omp unroll partial
        for (z = 0; z < k; z++) {
            t[z] = _mm512_setzero_pd(); // init t vector
        }

        for (int i = rows_load[tid]; i < rows_load[tid + 1]; i++) { // thread gets the row to process
            j = irp[i];
            lim = (irp[i+1] - j) / PD_STRIDE;

            for (iter = 0; iter < lim; iter++) {

                vals = _mm512_loadu_pd(&as[j]);                     // load 8 64-bit elements
                cols = _mm256_loadu_si256((__m256i*)&ja[j]);        // load 8 32-bit elements columns
                cols = _mm256_mullo_epi32(cols, scale);             // scale col index to be on the first column of x
                x_r = _mm512_i32gather_pd(cols, x, sizeof(Type));   // build 8 64-bit elements from x and cols
                t[0] = _mm512_fmadd_pd(vals, x_r, t[0]);            // execute a fused multiply-add

                for (z = 1; z < k; z++) {
                    cols = _mm256_add_epi32(cols, _MM8_1);
                    x_r = _mm512_i32gather_pd(cols, x, sizeof(Type));      // build 8 64-bit elements from x and cols
                    t[z] = _mm512_fmadd_pd(vals, x_r, t[z]);    // execute a fused multiply-add
                }

                j += PD_STRIDE;
            }

            r_y = i * k;

            // remainder loop if elements are not multiple of size
            for (; j < irp[i+1]; j++) {
                val = as[j];
                r_x = ja[j] * k;

                #pragma omp unroll partial
                for (z = 0; z < k; z++) {
                    y[r_y + z] += val * x[r_x + z];
                }
            }

            // reduce all 64-bit elements in t by addition
            #pragma omp unroll partial
            for (z = 0; z < k; z++) {
                y[r_y + z] += _mm512_reduce_add_pd(t[z]);
                t[z] = _mm512_setzero_pd();
            }
        }
    }
}

void spmm_csr_32(CSR *mat, const int* rows_load, int threads, const Type* x, int k, Type* y){

    int *irp = mat->IRP;
    int *ja = mat->JA;
    Type *as = mat->AS;

    const __m512i scale = _mm512_set1_epi32(k);

    #pragma omp parallel for num_threads(threads) shared(threads, rows_load, irp, k, as, ja, x, y, scale) default(none)
    for (int tid = 0; tid < threads; tid++) {   // parallelize on threads' id
        // private params
        int j, z, iter, lim, r_y, r_x;
        Type val;
        __m512i cols;
        __m512 vals, x_r;

        __m512 t[k];
        #pragma omp unroll partial
        for (z = 0; z < k; z++) { // init t vector
            t[z] = _mm512_setzero_ps();
        }

        for (int i = rows_load[tid]; i < rows_load[tid + 1]; i++) { // thread gets the row to process
            j = irp[i];
            lim = (irp[i+1] - j) / PS_STRIDE;

            for (iter = 0; iter < lim; iter++) {

                vals = _mm512_loadu_ps(&as[j]);                 // load 16 32-bit elements
                cols = _mm512_loadu_si512(&ja[j]);              // load 16 32-bit elements columns
                cols = _mm512_mullo_epi32(cols, scale);         // scale col index to be on the first column of x
                x_r = _mm512_i32gather_ps(cols, x, sizeof(Type));          // build 16 32-bit elements from x and cols
                t[0] = _mm512_fmadd_ps(vals, x_r, t[0]);        // execute a fused multiply-add

                for (z = 1; z < k; z++) {
                    cols = _mm512_add_epi32(cols, _MM16_1);         // shift col idx by 1 to shift x column
                    x_r = _mm512_i32gather_ps(cols, x, sizeof(Type));
                    t[z] = _mm512_fmadd_ps(vals, x_r, t[z]);
                }

                j += PS_STRIDE;
            }

            r_y = i * k;

            // remainder loop if elements are not multiple of size
            for (; j < irp[i+1]; j++) {
                val = as[j];
                r_x = ja[j] * k;

                #pragma omp unroll partial
                for (z = 0; z < k; z++) {
                    y[r_y + z] += val * x[r_x + z];
                }
            }

            // reduce all 32-bit elements in t by addition
            #pragma omp unroll partial
            for (z = 0; z < k; z++) {
                y[r_y + z] += _mm512_reduce_add_ps(t[z]);
                t[z] = _mm512_setzero_ps();
            }
        }
    }
}

void spmm_csr(CSR *mat, const int* rows_load, int threads, const Type* x, int k, Type* y){
    if (sizeof(Type) == 8) {
        spmm_csr_64(mat, rows_load, threads, x, k, y);
    } else {
        spmm_csr_32(mat, rows_load, threads, x, k, y);
    }
}

/**
 * Load balancing related to the amount of non-zeros given to each thread.
 * The number of non-zeros is always rounded to be contained in a full row to maintain locality.
 * */
void csr_nz_balancing(int threads, int tot_nz, const int* irp, int tot_rows, int* rows_idx){
    int j, nz, nz_prev = 0, nz_curr, start_row = 0, r_prev, r_curr = 0;

    for (int i = 0; i < threads; i++) {
        rows_idx[i] = start_row; // add the idx of the start row
        printf("."); // TODO: why this print enormously speed omp product???

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
    printf("\n");
}