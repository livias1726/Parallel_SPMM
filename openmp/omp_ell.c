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

void spmm_ell_vector(int *ja, Type *as, int start, int end, Type* x, int k, __m512d* t, Type* y) {

    int z;
    __m256i cols;
    __m512d vals, x_r;

    const __m256i scale = _MM8(k);
    const __m256i one = _MM8(1);

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

/**
 * Computes the product with A stored in a ELL format
 *
 * @param mat matrix in ellpack format
 * @param threads number of threads to spawn
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
/*
void spmm_ell(ELL* mat, int threads, Type* x, int k, Type* y) {
    int maxnz = mat->MAXNZ;
    int rows = mat->M;
    int *ja = mat->JA;
    Type *as = mat->AS;

    int vector = maxnz / PD_STRIDE;
    int j = PD_STRIDE * vector;
    int remainder = maxnz % PD_STRIDE;

    int start, end, r_y;

    __m512d t[k];
    #pragma ivdep
    #pragma omp unroll partial
    for (int z = 0; z < k; z++) t[z] = _mm512_setzero_pd(); // init t vector

#pragma omp parallel for num_threads(threads) private(start, end, r_y, t) \
                             shared(j, maxnz, rows, ja, as, k, x, y, vector, remainder) default(none)
    for (int i = 0; i < rows; ++i) { // thread takes the row
        start = i * maxnz;
        end = start + j;
        r_y = i * k;

        if (vector) { // the row can be vectorized at least once (maxnz > PD_STRIDE)
            spmm_ell_vector(ja, as, start, end, x, k, t, &y[r_y]);

            // there are remaining non-zeros that are not padding
            if (remainder && as[end]) {
                spmm_ell_stream(end, start + maxnz, ja, as, x, k, &y[r_y]);
            }

        } else {
            spmm_ell_stream(end, start + maxnz, ja, as, x, k, &y[r_y]);
        }
    }
}
*/

 void spmm_ell(ELL* mat, int* thread_rows, int* thread_maxnz, int threads, Type* x, int k, Type* y) {
     int *ja = mat->JA;
     Type *as = mat->AS;
     int maxnz = mat->MAXNZ;

     int stride = STRIDE;

     __m512d t[k];
#pragma ivdep
#pragma omp unroll partial
     for (int z = 0; z < k; z++) t[z] = _mm512_setzero_pd(); // init t vector

#pragma omp parallel for num_threads(threads) \
        private(t) \
        shared(threads, thread_rows, thread_maxnz, stride, maxnz, ja, as, k, x, y) \
        default(none)
     for (int tid = 0; tid < threads; ++tid) {

         int local_maxnz = thread_maxnz[tid];
         int num_vectors = local_maxnz / stride;
         int remainder = local_maxnz % stride;
         int j = stride * num_vectors;

         int start, end, r_y;

         for (int i = thread_rows[tid]; i < thread_rows[tid+1]; ++i) { // thread takes the row
             start = i * maxnz;
             end = start + j;
             r_y = i * k;


             if (num_vectors > 0) { // the row can be vectorized at least once (maxnz > PD_STRIDE)
                 spmm_ell_vector(ja, as, start, end, x, k, t, &y[r_y]);

                 // there are remaining non-zeros that are not padding
                 if (remainder && as[end] != 0) spmm_ell_stream(end, start + local_maxnz, ja, as, x, k, &y[r_y]);

             } else {
                 spmm_ell_stream(end, start + local_maxnz, ja, as, x, k, &y[r_y]);
             }
         }
     }
 }

int get_row_nz(Type* as, int start, int end){
    int ctr = 0;

    for(int i = start; i < start + end; ++i){
        if (as[i] == 0) break;
        ++ctr;
    }

    return ctr;
}

/**
 * Load balancing related to the amount of non-zeros given to each computational node.
 * The number of non-zeros is always rounded to be contained in a full row to maintain locality
 *
 * @param ts number of threads
 * */
 //TODO: manage rows < threads
void ell_nz_balancing(ELL* ell, int threads, int* thread_rows, int* thread_maxnz){
    // ELL
    int tot_rows = ell->M, tot_nz = ell->NZ, maxnz = ell->MAXNZ;
    Type *as = ell->AS;

    int j, nz, nz_prev = 0, nz_curr = 0;
    int r_nz, maxnz_curr = 0, maxnz_prev = 0; // non-zeros in the row

    int stride = STRIDE;
    int rem;

    thread_rows[0] = 0;
    for (int i = 1; i < threads; i++) {
        printf("."); // TODO: increases performances???

        // compute the number of non-zeros to assign the i-th thread
        nz = INT_LOAD_BALANCE(i, tot_nz, threads);

        for (j = thread_rows[i-1]; j < tot_rows; j++) { // scan the remaining rows
            // maxnz
            r_nz = get_row_nz(as, j * maxnz, maxnz); // get number of nz in  row
            if (r_nz > maxnz_curr) { // new local maxnz
                rem = r_nz % stride;
                if (rem && (maxnz - r_nz) >= rem) r_nz += rem; // can be padded to be a multiple of stride

                maxnz_curr = r_nz;
            }

            // nz
            nz_curr += r_nz;
            if (nz_curr >= nz) {
                // get the number of rows that includes a number of nz closer to the one assigned
                if ((nz - nz_prev) < (nz_curr - nz)) { // exclude current row
                    thread_rows[i] = j;
                    thread_maxnz[i-1] = maxnz_prev;
                } else { // include current row
                    thread_rows[i] = j + 1;
                    thread_maxnz[i-1] = maxnz_curr;
                }

                break;
            }

            nz_prev = nz_curr; // count of nz is still lower than the number of nz assigned to the thread
            maxnz_prev = maxnz_curr;
        }

        nz_prev = 0;
        nz_curr = 0;
        maxnz_prev = 0;
        maxnz_curr = 0;
    }

    // need to count maxnz for the last thread
    for (j = thread_rows[threads-1]; j < tot_rows; j++) {
        r_nz = get_row_nz(as, j * maxnz, maxnz); // get number of nz in the considered rows
        if (r_nz > maxnz_curr) { // new local maxnz
            rem = r_nz % stride;
            if (rem && (maxnz - r_nz) >= rem) r_nz += rem; // can be padded to be multiple of stride

            maxnz_curr = r_nz;
        }
    }

    thread_rows[threads] = tot_rows; // last thread gets the remaining rows
    thread_maxnz[threads-1] = maxnz_curr;

    printf("\n"); // to
}
