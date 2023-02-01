#include "utils.h"

void insert_in_row(Elem*** arr, Elem** node, int idx) {
    Elem* head = (*arr)[idx];

    if (head == NULL) {
        (*node)->nz = 1;
        (*arr)[idx] = *node;
    } else {
        Elem* prev = NULL;
        Elem* curr = head;
        while (curr != NULL) {
            prev = curr;
            curr = curr->next;
        }
        prev->next = *node;
        head->nz++;
    }
}

Elem** read_mm(FILE* f, int m, int* nz, const MM_typecode t){
    // array of lists of Elem: 1 per row
    Elem** elems = (Elem**) calloc(m, sizeof(Elem*));

    // scan matrix
    int r, c, i;
    for (i = 0; i < *nz; i++){
        Elem* elem = (Elem*)malloc(sizeof(Elem));
        if (mm_is_pattern(t)) {
            fscanf(f, "%d %d\n", &r, &c);
            elem->val = 1.0;
        } else {
            fscanf(f, "%d %d %lg\n", &r, &c, &(elem->val));
        }

        elem->j = --c;
        elem->next = NULL;
        insert_in_row(&elems, &elem, --r);

        if (mm_is_symmetric(t) && r != c) {
            Elem* elem_s = (Elem*)malloc(sizeof(Elem));
            elem_s->val = elem->val;
            elem_s->j = r;
            elem_s->next = NULL;
            insert_in_row(&elems, &elem_s, c);

            *nz += 1;
        }
    }

    return elems;
}

void read_mm_csr(FILE* f, CSR** mat, MM_typecode t){
    int ret, i, m, n, nz, elem_count = 0;
    Elem *curr, *prev;
    Elem** elems;

    // process the matrix size information
    ret = mm_read_mtx_crd_size(f, &m, &n, &nz);
    if (ret != 0) { exit(-1); }

    // read matrix from file
    elems = read_mm(f, m, &nz, t);

    // alloc memory
    *mat = (CSR*) malloc(sizeof(CSR));
    error_handler(*mat);

    (*mat)->IRP = (int*)malloc(m*sizeof(int));
    (*mat)->JA = (int*)malloc(nz*sizeof(int));
    (*mat)->AS = (double*)malloc(nz*sizeof(double));
    error_handler((*mat)->IRP);
    error_handler((*mat)->JA);
    error_handler((*mat)->AS);

    // populate CSR format
    (*mat)->M = m;
    (*mat)->N = n;
    // scan the array of lists: 1 per row
    for (i = 0; i < m; i++){
        curr = elems[i];

        // skip empty rows
        if (curr == NULL) { continue; }

        // update rows pointers
        (*mat)->IRP[i] = elem_count;

        // scan elements of i-th row and dealloc memory
        while (curr != NULL) {
            (*mat)->AS[elem_count] = curr->val;
            (*mat)->JA[elem_count] = curr->j;

            prev = curr;
            curr = curr->next;
            free(prev);
            elem_count++;
        }
    }

    free(elems);
}

void read_mm_ell(FILE* f, ELL** mat, MM_typecode t){
    int ret, i, m, n, nz, maxnz, count = 0;
    Elem *curr, *prev;
    Elem** elems;

    // process the matrix size information
    ret = mm_read_mtx_crd_size(f, &m, &n, &nz);
    if (ret != 0) { exit(-1); }

    // read matrix from file
    elems = read_mm(f, m, &nz, t);

    // retrieve maxnz
    maxnz = 0;
    for (i = 0; i < m; i++) {
        if ((elems[i] != NULL) && (maxnz < elems[i]->nz)) {
            maxnz = elems[i]->nz;
        }
    }

    // alloc memory:
    // calloc is used to avoid the addition of padding in a loop
    // 2D arrays are treated as 1D arrays
    (*mat)->JA = (int*)calloc((m-1)*(maxnz-1), sizeof(int));
    (*mat)->AS = (double*)calloc((m-1)*(maxnz-1), sizeof(double));
    error_handler((*mat)->JA);
    error_handler((*mat)->AS);

    // populate ELLPACK format
    (*mat)->M = m;
    (*mat)->N = n;
    (*mat)->MAXNZ = maxnz;

    // scan the array of lists: 1 per row
    for (i = 0; i < m; i++){
        curr = elems[i];

        // skip empty rows
        if (curr == NULL) { continue; }

        // scan elements of i-th row and dealloc memory
        while (curr != NULL) {
            (*mat)->JA[i*maxnz + count] = curr->j;
            (*mat)->AS[i*maxnz + count] = curr->val;

            prev = curr;
            curr = curr->next;
            free(prev);

            count++;
        }

        count = 0;
    }

    free(elems);
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