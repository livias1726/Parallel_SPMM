#include "headers/omp_csr.h"

/**
 * CSR SpMM with DOUBLE values using SIMD processing with
 * Intel's Advanced Vector Extensions (AVX) intrinsic functions.
 *
 * @param mat matrix in csr format
 * @param rows_load number of rows per thread
 * @param threads number of threads
 * @param x multivector stored as a 1D array
 * @param k number of columns of x
 * @param y receives product results stored as a 1D array
 * */
void spmm_csr_64(CSR *mat, const int* rows_load, int threads, const Type* x, int k, Type* y){

    int *irp = mat->IRP, *ja = mat->JA, m = mat->M;
    Type *as = mat->AS;

    const __m256i scale = _MM8(k);
    const __m256i one = _MM8(1);

    if (m < threads) threads = m - 1;

#pragma omp parallel num_threads(threads)
    {
        int j, z, end, r_y, r_x, tid = omp_get_thread_num();
        Type val;
        __m256i cols;
        __m512d vals, x_r, t[k];

        #pragma ivdep
        #pragma omp unroll partial
        for (z = 0; z < k; z++) t[z] = _mm512_setzero_pd(); // init t vector

        for (int i = rows_load[tid]; i < rows_load[tid + 1]; ++i) { // thread gets the row to process
            r_y = i * k;
            j = irp[i];
            end = (((irp[i + 1] - j) / PD_STRIDE) * PD_STRIDE) + j;

            if (j < end) {
                // perform a SIMD product on chunks of 8 non-zeros
                do{
                    vals = _mm512_loadu_pd(&as[j]);                     // load 8 64-bit elements
                    cols = _mm256_loadu_si256((__m256i * ) & ja[j]);        // load 8 32-bit elements columns
                    cols = _mm256_mullo_epi32(cols, scale);             // scale col index to be on the first column of x
                    x_r = _mm512_i32gather_pd(cols, x, sizeof(Type));   // build 8 64-bit elements from x and cols
                    t[0] = _mm512_fmadd_pd(vals, x_r, t[0]);            // execute a fused multiply-add

                    for (z = 1; z < k; z++) {
                        cols = _mm256_add_epi32(cols, one);
                        x_r = _mm512_i32gather_pd(cols, x, sizeof(Type));      // build 8 64-bit elements from x and cols
                        t[z] = _mm512_fmadd_pd(vals, x_r, t[z]);    // execute a fused multiply-add
                    }

                    j += PD_STRIDE;

                }while(j < end);

                // reduce all 64-bit elements in t by addition
                #pragma ivdep
                #pragma omp unroll partial
                for (z = 0; z < k; z++) {
                    y[r_y + z] += _mm512_reduce_add_pd(t[z]);
                    t[z] = _mm512_setzero_pd();
                }
            }

            // remainder loop if elements are not multiple of size
            for (; j < irp[i + 1]; j++) {
                val = as[j];
                r_x = ja[j] * k;

                #pragma ivdep
                #pragma omp unroll partial
                for (z = 0; z < k; z++) y[r_y + z] += val * x[r_x + z];
            }
        }
    }
}

/**
 * CSR SpMM with FLOAT values using SIMD processing with
 * Intel's Advanced Vector Extensions (AVX) intrinsic functions.
 *
 * @param mat matrix in csr format
 * @param rows_load number of rows per thread
 * @param threads number of threads
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void spmm_csr_32(CSR *mat, const int* rows_load, int threads, const Type* x, int k, Type* y){

    int *irp = mat->IRP, *ja = mat->JA, m = mat->M;
    Type *as = mat->AS;

    const __m512i scale = _MM16(k);
    const __m512i one = _MM16(1);

    if (m < threads) threads = m - 1;

#pragma omp parallel num_threads(threads)
    {
        int j, z, end, r_y, r_x, tid = omp_get_thread_num();
        Type val;
        __m512i cols;
        __m512 vals, x_r, t[k];

        #pragma ivdep
        #pragma omp unroll partial
        for (z = 0; z < k; z++) t[z] = _mm512_setzero_ps();

        for (int i = rows_load[tid]; i < rows_load[tid + 1]; ++i) { // thread gets the row to process
            r_y = i * k;
            j = irp[i];
            end = (((irp[i + 1] - j) / PS_STRIDE) * PS_STRIDE) + j;

            if (j < end) {
                // perform a SIMD product on chunks of 16 non-zeros
                do {
                    vals = _mm512_loadu_ps(&as[j]);                 // load 16 32-bit elements
                    cols = _mm512_loadu_si512(&ja[j]);              // load 16 32-bit elements columns
                    cols = _mm512_mullo_epi32(cols, scale);         // scale col index to be on the first column of x
                    x_r = _mm512_i32gather_ps(cols, x,
                                              sizeof(Type));          // build 16 32-bit elements from x and cols
                    t[0] = _mm512_fmadd_ps(vals, x_r, t[0]);        // execute a fused multiply-add

                    for (z = 1; z < k; z++) {
                        cols = _mm512_add_epi32(cols, one);         // shift col idx by 1 to shift x column
                        x_r = _mm512_i32gather_ps(cols, x, sizeof(Type));
                        t[z] = _mm512_fmadd_ps(vals, x_r, t[z]);
                    }

                    j += PS_STRIDE;

                } while(j < end);

                // reduce all 32-bit elements in t by addition
                #pragma ivdep
                #pragma omp unroll partial
                for (z = 0; z < k; z++) {
                    y[r_y + z] += _mm512_reduce_add_ps(t[z]);
                    t[z] = _mm512_setzero_ps();
                }
            }

            // remainder loop if elements are not multiple of size
            for (; j < irp[i+1]; j++) {
                val = as[j];
                r_x = ja[j] * k;

                #pragma ivdep
                #pragma omp unroll partial
                for (z = 0; z < k; z++) y[r_y + z] += val * x[r_x + z];
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
 * The number of non-zeros is always rounded to be contained in a full row.
 * */
void csr_nz_balancing(int threads, int tot_nz, const int* irp, int tot_rows, int* rows_idx){
    int j, nz, nz_prev, nz_curr;

    if (tot_rows < threads) threads = tot_rows - 1;

    rows_idx[0] = 0;
    for (int i = 1; i < threads; i++) {
        printf("."); // TODO: why this print enormously speed omp product???

        // compute the number of non-zeros to assign the i-th thread
        nz = INT_LOAD_BALANCE(i, tot_nz, threads);

        nz_curr = 0;
        for (j = rows_idx[i-1]; j < tot_rows; j++) {
            nz_curr += irp[j + 1] - irp[j]; // get number of nz in the considered rows

            if (nz_curr >= nz) {
                // get the number of rows that includes a number of nz closer to the one assigned
                rows_idx[i] = ((nz_curr - nz) < (nz - nz_prev)) ? j + 1 : j;
                break;
            }

            nz_prev = nz_curr; // count of nz is still lower than the number of nz assigned to the thread
        }
    }

    rows_idx[threads] = tot_rows; // last thread gets the remaining rows
    printf("\n");
}