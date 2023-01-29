#include "utils.h"

void quicksort(Elem** elems, int first, int last) {
    int i, j, pivot;
    Elem temp;

    if(first < last){
        pivot = first;
        i = first;
        j = last;

        while(i < j){
            while(elems[i]->i <= elems[pivot]->i && i < last) { i++; }
            while(elems[j]->i > elems[pivot]->i) { j--; }
            if(i < j){
                temp = *elems[i];
                *elems[i] = *elems[j];
                *elems[j] = temp;
            }
        }

        temp = *elems[pivot];
        *elems[pivot] = *elems[j];
        *elems[j] = temp;

        quicksort(elems, first, j-1);
        quicksort(elems, j+1, last);
    }
}

void expand_symmetry(Elem **elems, int *nz, int num_diag) {
    int i, idx, new_nz = 2*(*nz)-num_diag;
    *elems = (Elem*) realloc(*elems, new_nz*sizeof(Elem));

    idx = 0;
    for (i = 0; i < *nz; i++) {
        if ((*elems)[i].i == (*elems)[i].j) {
            continue;
        } else {
            (*elems)[*nz+idx].i = (*elems)[i].j;
            (*elems)[*nz+idx].j = (*elems)[i].i;
            (*elems)[*nz+idx].val = (*elems)[i].val;

            idx++;
        }
    }

    *nz = new_nz;
}

Elem* read_mm(FILE* f, int* nz, const MM_typecode t){
    // temporary element array
    Elem* elems = (Elem*) malloc(*nz*sizeof(Elem));

    // scan matrix
    int diag=0;
    for (int i=0; i < *nz; i++){
        if (mm_is_pattern(t)) {
            fscanf(f, "%d %d\n", &(elems[i].i), &(elems[i].j));
            elems[i].val = 1.0;
        } else {
            fscanf(f, "%d %d %lg\n", &(elems[i].i), &(elems[i].j), &(elems[i].val));
        }

        if (mm_is_symmetric(t) && elems[i].i==elems[i].j) {
            diag++;
        }

        // adjust from 1-based to 0-based
        elems[i].i--;
        elems[i].j--;
    }

    if (mm_is_symmetric(t)) {
        expand_symmetry(&elems, nz, diag);
    }

    quicksort(&elems, 0, *nz-1);

    return elems;
}

void read_mm_csr(FILE* f, CSR** mat, MM_typecode t){
    int ret, i, m, n, nz;

    // skip the optional comments and process the matrix size information
    ret = mm_read_mtx_crd_size(f, &m, &n, &nz);
    if (ret != 0) {
        exit(-1);
    }

    Elem* elem = read_mm(f, &nz, t);

    // alloc memory
    *mat = (CSR*) malloc(sizeof(CSR));
    error_handler(*mat);

    (*mat)->IRP = (int*)malloc(m*sizeof(int));
    (*mat)->JA = (int*)malloc(nz*sizeof(int));
    (*mat)->AS = (double*)malloc(nz*sizeof(double));
    error_handler((*mat)->IRP);
    error_handler((*mat)->JA);
    error_handler((*mat)->AS);

    (*mat)->M = m;
    (*mat)->N = n;
    for (i = 0; i < nz; i++){
        (*mat)->AS[i] = elem[i].val;
        (*mat)->JA[i] = elem[i].j;
        (*mat)->IRP[elem[i].i + 1]++;
    }

    for (i = 0; i < m; i++){
        (*mat)->IRP[i + 1] += (*mat)->IRP[i];
    }

    free(elem);
}

void read_mm_ell(FILE* f, ELL** mat, MM_typecode t){
    int ret, i, m, n, nz, maxnz, count, r;

    // skip the optional comments and process the matrix size information
    ret = mm_read_mtx_crd_size(f, &m, &n, &nz);
    if (ret != 0) {
        exit(-1);
    }

    Elem* elem = read_mm(f, &nz, t);

    // compute maxnz
    maxnz = 1;
    count = 1;
    r = elem[0].i;
    for (i = 1; i < nz; i++) {
        if (r == elem[i].i) {
            count++;
        } else {
            maxnz = count > maxnz ? count : maxnz;
            count = 0;
        }
    }

    // alloc memory
    (*mat)->JA = (int*)malloc((m-1)*(maxnz-1)*sizeof(int));
    (*mat)->AS = (double*)malloc((m-1)*(maxnz-1)*sizeof(double));
    error_handler((*mat)->JA);
    error_handler((*mat)->AS);

    (*mat)->M = m;
    (*mat)->N = n;
    (*mat)->MAXNZ = maxnz;

    int r2, r1=0, c=0;
    for (i = 0; i < nz; i++){
        r2 = elem[i].i-1;
        if(r1 != r2){
            while (c != maxnz-1) { // add padding
                (*mat)->AS[r1*maxnz+c] = 0;
                (*mat)->JA[r1*maxnz+c] = 0;
                c++;
            }
            c=0;
        }

        (*mat)->AS[r2*maxnz+c] = elem[i].val;
        (*mat)->JA[r2*maxnz+c] = elem[i].j;
        c++;
    }

    free(elem);
}

void check_mat_type(MM_typecode t) {
    if ((!mm_is_real(t) || !mm_is_pattern(t)) && !mm_is_sparse(t)) {
        printf("This application does not support Market Market type: [%s]\n", mm_typecode_to_str(t));
        exit(-1);
    }
}

/**
 * Computes the MFLOPS
 *
 * TODO: change accordingly to the application
 * */
void get_mflops(time_t v, int dims){
    int num_ops = 2*dims*dims*dims;
    printf("MFLOPS: %f\n\n", num_ops/((double)v*pow(10,6)));
}


void error_handler(void *p) {
    if(p == NULL){
        fprintf(stderr, "Malloc failed");
        exit(-1);
    }
}