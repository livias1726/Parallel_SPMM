#include "headers/omp_ell.h"

void spmm_ell_stream(int start, int end, const int *ja, const Type *as, Type* x, int k, Type* y){

    int z;
    Type val, *x_r;

    for (int i = start; i < end; ++i) {
        val = as[i];
        if (val == 0) break; // if padding is reached break loop

        x_r = &x[ja[i] * k]; // prefetching of the row in x to read from

        #pragma ivdep
        #pragma omp unroll partial
        for (z = 0; z < k; z++) y[z] += val * x_r[z];
    }
}

void spmm_ell_vector_64(int *ja, Type *as, int start, int end, Type* x, int k, Type* y) {

    int z;
    __m256i cols;
    __m512d vals, x_r;

    const __m256i scale = MM8(k);
    const __m256i one = MM8(1);

    __m512d t[k];
#pragma ivdep
#pragma omp unroll partial
    for (z = 0; z < k; z++) t[z] = _mm512_setzero_pd(); // init t vector

    #pragma ivdep
    #pragma omp unroll partial
    for (int idx = start; idx < end; idx += PD_STRIDE) { // vectorization of 8 elements per iteration
        if (as[idx] == 0) break;    // break loop if padding reached

        vals = _mm512_loadu_pd(&as[idx]);                     // load 8 64-bit elements
        cols = _mm256_loadu_si256((__m256i*)&ja[idx]);        // load 8 32-bit elements columns
        cols = _mm256_mullo_epi32(cols, scale);             // scale col index to be on the first column of x
        x_r = _mm512_i32gather_pd(cols, x, sizeof(Type));   // build 8 64-bit elements from x and cols
        t[0] = _mm512_fmadd_pd(vals, x_r, t[0]);            // execute a fused multiply-add

        for (z = 1; z < k; z++) {
            cols = _mm256_add_epi32(cols, one);
            x_r = _mm512_i32gather_pd(cols, x, sizeof(Type));      // build 8 64-bit elements from x and cols
            t[z] = _mm512_fmadd_pd(vals, x_r, t[z]);    // execute a fused multiply-add
        }
    }

    #pragma ivdep
    #pragma omp unroll partial
    for (z = 0; z < k; z++) {
        y[z] += _mm512_reduce_add_pd(t[z]);
        t[z] = _mm512_setzero_pd();
    }
}

void spmm_ell_vector_32(int *ja, Type *as, int start, int end, Type* x, int k, Type* y) {

    int z;
    __m512i cols;
    __m512 vals, x_r;

    const __m512i scale = MM16(k);
    const __m512i one = MM16(1);

    __m512 t[k];
#pragma ivdep
#pragma omp unroll partial
    for (z = 0; z < k; z++) t[z] = _mm512_setzero_ps(); // init t vector

    #pragma ivdep
    #pragma omp unroll partial
    for (int idx = start; idx < end; idx += PS_STRIDE) { // vectorization of 16 elements per iteration
        if (as[idx] == 0) break;    // break loop if padding reached

        vals = _mm512_loadu_ps(&as[idx]);                 // load 16 32-bit elements
        cols = _mm512_loadu_si512(&ja[idx]);              // load 16 32-bit elements columns
        cols = _mm512_mullo_epi32(cols, scale);         // scale col index to be on the first column of x
        x_r = _mm512_i32gather_ps(cols, x, sizeof(Type));          // build 16 32-bit elements from x and cols
        t[0] = _mm512_fmadd_ps(vals, x_r, t[0]);        // execute a fused multiply-add

        for (z = 1; z < k; z++) {
            cols = _mm512_add_epi32(cols, one);         // shift col idx by 1 to shift x column
            x_r = _mm512_i32gather_ps(cols, x, sizeof(Type));
            t[z] = _mm512_fmadd_ps(vals, x_r, t[z]);
        }
    }

#pragma ivdep
#pragma omp unroll partial
    for (z = 0; z < k; z++) {
        y[z] += _mm512_reduce_add_ps(t[z]);
        t[z] = _mm512_setzero_ps();
    }
}

/**
 * Computes the product with A stored in a ELL format
 *
 * @param mat matrix in ellpack format
 * @param threads number of threads to spawn
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void spmm_ell(ELL* mat, int threads, Type* x, int k, Type* y) {

    int maxnz = mat->MAXNZ, rows = mat->M, *ja = mat->JA;
    Type *as = mat->AS;

    int stride = STRIDE;

    int vector = maxnz / stride;
    int j = stride * vector;
    int remainder = maxnz % stride;

    int start, end, r_y;

#pragma omp parallel for num_threads(threads) private(start, end, r_y) \
                             shared(j, maxnz, rows, ja, as, k, x, y, vector, remainder) default(none)
    for (int i = 0; i < rows; ++i) { // thread takes the row
        start = i * maxnz;
        end = start + j;
        r_y = i * k;

        memset(&y[r_y], 0, k * sizeof(Type));

        if (vector) { // the row can be vectorized at least once
            if (sizeof(Type) == 8) {
                spmm_ell_vector_64(ja, as, start, end, x, k, &y[r_y]);
            } else {
                spmm_ell_vector_32(ja, as, start, end, x, k, &y[r_y]);
            }

            if (remainder && as[end]) { // there are remaining non-zeros that are not padding
                spmm_ell_stream(end, start + maxnz, ja, as, x, k, &y[r_y]);
            }
        } else {
            spmm_ell_stream(end, start + maxnz, ja, as, x, k, &y[r_y]);
        }
    }
}
