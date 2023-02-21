#include "omp_utils.h"

/**
 * Read the matrix into a CSR struct representing the matrix in CSR storage format
 *
 * @param f file descriptor
 * @param t matrix type code
 * */
CSR* read_mm_csr(FILE* f, MM_typecode t){
    // original variables
    int i, m, n, nz;
    Elem *curr, *prev;
    // parallel variables
    int idx, nz_add;

    // read matrix from file
    Elem** elems = read_mm(f, &m, &n, &nz, t);
    // alloc memory
    CSR* mat = alloc_csr(m, n, nz);

    // used to avoid critical section coordination in parallel execution
    int elem_counts[m];
    // TODO: check if it is better like this or with critical sections in openmp block
    for (i = 0; i < m; i++) {
        curr = elems[i];
        nz_add = (curr == NULL) ? 0 : curr->nz;
        elem_counts[i] = (i == 0) ? nz_add : elem_counts[i-1] + nz_add;
    }

    // scan the array of lists: 1 per row
    /* A static schedule is non-optimal when the different iterations take different amounts of time*/
#pragma omp parallel for schedule(guided) \
            shared(m, elems, mat, elem_counts) \
            private(idx, curr, prev) \
            default(none)
    for (i = 0; i < m; i++){
        curr = elems[i];
        // skip empty rows
        if (curr == NULL) { continue; }

        // update rows pointers
        mat->IRP[i] = (i == 0) ? 0 : elem_counts[i-1];

        // scan elements of i-th row and dealloc memory
        // range of action per thread: [elem_counts[i-1], elem_count[i])
        idx = mat->IRP[i];
        while (curr != NULL && idx < elem_counts[i]) {
            mat->AS[idx] = curr->val;
            mat->JA[idx] = curr->j;

            prev = curr;
            curr = curr->next;
            free(prev);

            idx++;
        }
    }

    free(elems);
    return mat;
}

/**
 * Read the matrix into a ELL struct representing the matrix in ELLPACK storage format
 *
 * @param f file descriptor
 * @param t matrix type code
 * */
ELL* read_mm_ell(FILE* f, MM_typecode t){
    int i, m, n, nz, maxnz, count = 0;
    Elem *curr, *prev;

    // read matrix from file
    Elem** elems = read_mm(f, &m, &n, &nz, t);
    // alloc memory
    ELL* mat = alloc_ell(elems, m, n, nz, &maxnz);

    // scan the array of lists: 1 per row
#pragma omp parallel for schedule(guided) \
            shared(m, maxnz, elems, mat) \
            private(count, curr, prev) \
            default(none)
    for (i = 0; i < m; i++){
        curr = elems[i];

        // scan elements of i-th row and dealloc memory
        while (curr != NULL) {
            mat->JA[i*maxnz + count] = curr->j;
            mat->AS[i*maxnz + count] = curr->val;

            prev = curr;
            curr = curr->next;
            free(prev);

            count++;
        }

        count = 0;
    }

    free(elems);
    return mat;
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
int* nz_balancing(int ts, int tot_nz, const int* irp, int tot_rows){
    int i, j, r1, nz, start_row = 0, r2 = 0;

    int* nz_start = (int*) malloc(ts* sizeof(int));
    malloc_handler(1, (void*[]){nz_start}, 147);

    for (i = 0; i < ts; i++) {
        nz_start[i] = start_row; // add the idx of the start row
        if (i == ts-1) { // if last thread, get the remaining rows
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

    return nz_start;
}