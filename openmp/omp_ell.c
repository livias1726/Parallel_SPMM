#include "headers/omp_ell.h"

void spmm_ell_stream(int start, int end, int *ja, Type *as, Type* x, int k, Type* y){

    int z;
    Type val, *x_r;

    for (int i = start; i < end; ++i) {
        val = as[i];
        if (val == 0) break; // if padding is reached break loop

        x_r = &x[ja[i] * k]; // prefetching of the row in x to read from

        #pragma ivdep
        #pragma omp unroll partial
        for (z = 0; z < k; z++) {
            y[z] += val * x_r[z];
        }
    }
}

void spmm_ell_vector(int *ja, Type *as, int start, int end, Type* x, int k, Type* y) {

    int z;
    __m256i cols;
    __m512d vals, x_r;

    __m512d t[k];
    #pragma ivdep
    #pragma omp unroll partial
    for (z = 0; z < k; z++) t[z] = _mm512_setzero_pd(); // init t vector

    const __m256i scale = _MM8(k);
    const __m256i one = _MM8(1);

    #pragma ivdep
    #pragma omp unroll partial
    for (int idx = start; idx < end; idx += PD_STRIDE) { // vectorization of 8 elements per iteration
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

/**
 * Computes the product with A stored in a ELL format
 *
 * @param mat matrix in ellpack format
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void spmm_ell(ELL* mat, int threads, Type* x, int k, Type* y) {
    int maxnz = mat->MAXNZ;
    int rows = mat->M;
    int *ja = mat->JA;
    Type *as = mat->AS;

    int vector = maxnz / PD_STRIDE;
    int j = PD_STRIDE * vector;
    int remainder = maxnz % PD_STRIDE;

    int start, end, r_y;

    #pragma omp parallel for num_threads(threads) private(start, end, r_y) \
                             shared(j, maxnz, rows, ja, as, k, x, y, vector, remainder) default(none)
    for (int i = 0; i < rows; ++i) { // thread takes the row
        start = i * maxnz;
        end = start + j;
        r_y = i * k;

        if (vector) { // the row can be vectorized at least once (maxnz > PD_STRIDE)
            spmm_ell_vector(ja, as, start, end, x, k, &y[r_y]);

            // there are remaining non-zeros that are not padding
            if (remainder && as[end]) {
                spmm_ell_stream(end, start + maxnz, ja, as, x, k, &y[r_y]);
            }

        } else {
            spmm_ell_stream(end, start + maxnz, ja, as, x, k, &y[r_y]);
        }
    }
}

// TODO: ???
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
