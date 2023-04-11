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
 * Read the matrix from .mtx file in coordinate format into an array of lists per row, to overcome
 * the column-wise order in which it is represented in the file.
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

        if (mm_is_pattern(t)) { // matrix is of type pattern
            fscanf(f, "%d %d\n", &r, &c);
            elem->val = 1.0;
        } else {
            if (sizeof(Type) == 8) {
                fscanf(f, "%d %d %lf\n", &r, &c, &(elem->val));
            } else {
                fscanf(f, "%d %d %f\n", &r, &c, &(elem->val));
            }

            // some matrices still have zero values in their representation: this condition avoids
#ifdef ELLPACK
            if (elem->val == 0) {
                free(elem);
                *nz -= 1;
                continue;
            }
#endif
        }

        elem->j = --c;
        elem->next = NULL;
        insert_in_row(elems, elem, --r);

        if (mm_is_symmetric(t) && r != c) {    // matrix is symmetric: re-build explicitly the upper triangle
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
 * Read the matrix into a CSR struct representing the matrix in CSR storage format.
 *
 * @param elems list of rows
 * @param m     number of rows
 * @param n     number of columns
 * @param nz    number of non-zeros
 * */
CSR* read_mm_csr(Elem** elems, int m, int n, int nz){
    int i, elem_count = 0;
    Elem *curr, *prev;

    // alloc memory
    CSR* mat = alloc_csr(m, n, nz);

    // scan the array of lists: 1 per row
    for (i = 0; i < m; i++){

        // update rows pointers
        mat->IRP[i] = elem_count;

        curr = elems[i];

        // skip empty rows
        if (curr == NULL) continue;

        // scan elements of i-th row and dealloc memory
        while (curr != NULL) {
            mat->AS[elem_count] = curr->val;
            mat->JA[elem_count] = curr->j;

            prev = curr;
            curr = curr->next;
            free(prev);

            ++elem_count;
        }
    }

    mat->IRP[m] = nz;

    free(elems);
    return mat;
}

/* Get MAXNZ from the thread's hack */
int get_maxnz(Elem** elems, int start, int end){
    int maxnz = 0;

    while ((start < end) && (elems[start] != NULL)) {
        maxnz = MAX(maxnz, elems[start]->nz);
        ++start;
    }

    return maxnz;
}

/**
 * Read the matrix into a ELL struct representing the matrix in ELLPACK storage format.
 *
 * @param elems list of rows
 * @param m     number of rows
 * @param n     number of columns
 * @param nz    number of non-zeros
 * */
ELL* read_mm_ell(Elem** elems, int m, int n, int nz){
    int maxnz, r, idx, count = 0;
    Elem *curr, *prev;

    // alloc memory
    maxnz = get_maxnz(elems, 0, m);
    ELL* mat = alloc_ell(m, n, nz, maxnz);

    // scan the array of lists: 1 per row
    for (int i = 0; i < m; i++){
        curr = elems[i];
        r = i * maxnz;

        // scan elements of i-th row and dealloc memory
        while (curr != NULL) {
            idx = r + count;

            mat->JA[idx] = curr->j;
            mat->AS[idx] = curr->val;

            prev = curr;
            curr = curr->next;
            free(prev);

            ++count;
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

    mat->IRP = (int*)malloc((m + 1) * sizeof(int));
    mat->JA = (int*)malloc(nz * sizeof(int));
    mat->AS = (Type*)malloc(nz * sizeof(Type));
    malloc_handler(3, (void* []) {mat->IRP, mat->JA, mat->AS});

    // populate CSR format
    mat->M = m;
    mat->N = n;
    mat->NZ = nz;

    return mat;
}

ELL* alloc_ell(int m, int n, int nz, int maxnz){
    // alloc memory
    ELL* mat = (ELL*) malloc(sizeof(ELL));
    malloc_handler(1, (void* []) {mat});

    // calloc is used to avoid the addition of padding in a loop
    int size = m * maxnz;
    mat->JA = (int *)calloc(size, sizeof(int));
    mat->AS = (Type *)calloc(size, sizeof(Type));
    malloc_handler(2, (void* []) {mat->JA, mat->AS});

    // populate ELLPACK format
    mat->M = m;
    mat->N = n;
    mat->NZ = nz;
    mat->MAXNZ = maxnz;

    return mat;
}
