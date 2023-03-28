#include "headers/utils.h"

/* adds an element representing a non-zero read from file to the list of the respective row */
void insert_in_row(Elem** arr, Elem* node, int idx) {
    Elem* head = arr[idx];

    if (head == NULL) {
        node->nz = 1;
        arr[idx] = node;
    } else {
        Elem* prev = NULL;
        Elem* curr = head;
        while (curr != NULL) {
            prev = curr;
            curr = curr->next;
        }
        prev->next = node;
        head->nz++;
    }
}

/**
 * Read the matrix from .mat files in coordinate format into an array of lists per row
 *
 * @param f file descriptor
 * @param m number of rows
 * @param nz pointer to the number of non-zeros (to be eventually updated)
 * @param t matrix type code
 * */
Elem** read_mm(FILE* f, int* m, int* n, int* nz, const MM_typecode t){
    int r, c, i, onz;

    // process the matrix size information
    if (mm_read_mtx_crd_size(f, m, n, nz) != 0) { exit(-1); }
    onz = *nz;

    // array of lists of Elem: 1 per row
    Elem** elems = (Elem**) malloc((*m)* sizeof(Elem*));
    malloc_handler(1, (void* []) {elems});
    for (i = 0; i < *m; i++) { elems[i] = NULL; }

    // scan matrix
    for (i = 0; i < onz; i++){
        Elem* elem = (Elem*)malloc(sizeof(Elem));
        malloc_handler(1, (void* []) {elem});

        if (mm_is_pattern(t)) {
            fscanf(f, "%d %d\n", &r, &c);
            elem->val = 1.0;
        } else {
            if (SP) {
                fscanf(f, "%d %d %f\n", &r, &c, &(elem->val));  // read single precision
            } else {
                fscanf(f, "%d %d %lf\n", &r, &c, &(elem->val)); // read double precision
            }

            // some matrices still have zero values in their representation: this condition excludes them
            if (elem->val == 0) {
                free(elem);
                *nz -= 1;
                continue;
            }
        }

        elem->j = --c;
        elem->next = NULL;
        insert_in_row(elems, elem, --r);

        if (mm_is_symmetric(t) && r != c) {
            Elem* elem_s = (Elem*)malloc(sizeof(Elem));
            malloc_handler(1, (void* []) {elem});

            elem_s->val = elem->val;
            elem_s->j = r;
            elem_s->next = NULL;
            insert_in_row(elems, elem_s, c);

            *nz += 1;
        }
    }

    return elems;
}

/**
 * Read the matrix into a CSR struct representing the matrix in CSR storage format
 * */
CSR* read_mm_csr(Elem** elems, int m, int n, int nz){
    int i, elem_count = 0;
    Elem *curr, *prev;

    // alloc memory
    CSR* mat = alloc_csr(m, n, nz);

    // scan the array of lists: 1 per row
    for (i = 0; i < m; i++){
        curr = elems[i];

        // skip empty rows
        if (curr == NULL) { continue; }

        // update rows pointers
        mat->IRP[i] = elem_count;

        // scan elements of i-th row and dealloc memory
        while (curr != NULL) {
            mat->AS[elem_count] = curr->val;
            mat->JA[elem_count] = curr->j;

            prev = curr;
            curr = curr->next;
            free(prev);
            elem_count++;
        }
    }

    mat->IRP[m] = nz;

    free(elems);
    return mat;
}

/**
 * Read the matrix into a ELL struct representing the matrix in ELLPACK storage format
 * */
ELL* read_mm_ell(Elem** elems, int m, int n, int nz){
    int i, maxnz, count = 0;
    Elem *curr, *prev;

    // alloc memory
    ELL* mat = alloc_ell(elems, m, n, nz, &maxnz);

    // scan the array of lists: 1 per row
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

CSR* alloc_csr(int m, int n, int nz){
    // alloc memory
    CSR* mat = (CSR*) malloc(sizeof(CSR));
    malloc_handler(1, (void* []) {mat});

    mat->IRP = (int*)malloc((m+1)*sizeof(int));
    mat->JA = (int*)malloc(nz*sizeof(int));
    mat->AS = (Type*)malloc(nz*sizeof(Type));
    malloc_handler(3, (void* []) {mat->IRP, mat->JA, mat->AS});

    // populate CSR format
    mat->M = m;
    mat->N = n;
    mat->NZ = nz;

    return mat;
}

ELL* alloc_ell(Elem** elems, int m, int n, int nz, int* maxnz){
    // retrieve maxnz
    *maxnz = 0;
    for (int i = 0; i < m; i++) {
        if ((elems[i] != NULL) && (*maxnz < elems[i]->nz)) {
            *maxnz = elems[i]->nz;
        }
    }

    // alloc memory
    ELL* mat = (ELL*) malloc(sizeof(ELL));
    malloc_handler(1, (void* []) {mat});
    // calloc is used to avoid the addition of padding in a loop
    // 2D arrays are treated as 1D arrays
    mat->JA = calloc(m*(*maxnz), sizeof(int));
    mat->AS = (Type*)calloc(m*(*maxnz), sizeof(Type));
    malloc_handler(2, (void* []) {mat->JA, mat->AS});

    // populate ELLPACK format
    mat->M = m;
    mat->N = n;
    mat->NZ = nz;
    mat->MAXNZ = *maxnz;

    return mat;
}