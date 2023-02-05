#include "utils.h"

//---------------------------------------------------------------------------------------Storage
/**
 * Add an element representing a non-zero read from file to the list
 * of the respective row
 *
 * @param arr array of list heads
 * @param node pointer to the new element
 * @param idx index of the row of the element
 * */
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
void read_mm(FILE* f, Elem** elems, int m, int* nz, const MM_typecode t){
    int r, c, i;

    // array of lists of Elem: 1 per row
    for (i = 0; i < m; i++) { elems[i] = NULL; }

    // scan matrix
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
        insert_in_row(elems, elem, --r);

        if (mm_is_symmetric(t) && r != c) {
            Elem* elem_s = (Elem*)malloc(sizeof(Elem));
            elem_s->val = elem->val;
            elem_s->j = r;
            elem_s->next = NULL;
            insert_in_row(elems, elem_s, c);

            *nz += 1;
        }
    }
}

/**
 * Read the matrix into a CSR struct representing the matrix in CSR storage format
 *
 * @param f file descriptor
 * @param t matrix type code
 * */
CSR* read_mm_csr(FILE* f, MM_typecode t){
    int i, m, n, nz, elem_count = 0;
    Elem *curr, *prev;
    CSR* mat;

    // process the matrix size information
    if (mm_read_mtx_crd_size(f, &m, &n, &nz) != 0) { exit(-1); }

    // read matrix from file
    Elem** elems = (Elem**) malloc(m* sizeof(Elem*));
    read_mm(f, elems, m, &nz, t);

    // alloc memory
    mat = (CSR*) malloc(sizeof(CSR));
    malloc_handler(1, (void* []) {mat}, 90);

    mat->IRP = (int*)malloc(m*sizeof(int));
    mat->JA = (int*)malloc(nz*sizeof(int));
    mat->AS = (double*)malloc(nz*sizeof(double));
    malloc_handler(3, (void* []) {mat->IRP, mat->JA, mat->AS}, 95);

    // populate CSR format
    mat->M = m;
    mat->N = n;
    mat->NZ = nz;
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
    ELL* mat;

    // process the matrix size information
    if (mm_read_mtx_crd_size(f, &m, &n, &nz) != 0) { exit(-1); }

    // read matrix from file
    Elem** elems = (Elem**) malloc(m* sizeof(Elem*));
    read_mm(f, elems, m, &nz, t);

    // retrieve maxnz
    maxnz = 0;
    for (i = 0; i < m; i++) {
        if ((elems[i] != NULL) && (maxnz < elems[i]->nz)) {
            maxnz = elems[i]->nz;
        }
    }

    // alloc memory
    mat = (ELL*) malloc(sizeof(ELL));
    malloc_handler(1, (void* []) {mat}, 155);
    // calloc is used to avoid the addition of padding in a loop
    // 2D arrays are treated as 1D arrays
    mat->JA = calloc(m*maxnz, sizeof(int));
    mat->AS = (double*)calloc(m*maxnz, sizeof(double));
    malloc_handler(2, (void* []) {mat->JA, mat->AS}, 160);

    // populate ELLPACK format
    mat->M = m;
    mat->N = n;
    mat->NZ = nz;
    mat->MAXNZ = maxnz;

    // scan the array of lists: 1 per row
    for (i = 0; i < m; i++){
        curr = elems[i];

        // scan elements of i-th row and dealloc memory
        while (curr != NULL) {
            mat->JA[i*maxnz + count] = curr->j; //TODO: fix segfault for bigger matrices
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

//------------------------------------------------------------------------------------------Others
void check_mat_type(MM_typecode t) {
    if ((!mm_is_real(t) || !mm_is_pattern(t)) && !mm_is_sparse(t)) {
        printf("This application does not support Market Market type: [%s]\n", mm_typecode_to_str(t));
        exit(-1);
    }
}

void populate_multivector(double* vec, int rows, int cols) {
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            vec[i*cols+j] = ((double)rand()/RAND_MAX);
        }
    }
}

void alloc_struct(double** vec, int rows, int cols) {
    *vec = (double*) malloc(rows*cols* sizeof(double));
    malloc_handler(1, (void*[]){*vec}, 209);
}

void malloc_handler(int size, void **p, int line) {
    for(int i=0; i<size; i++){
        if(p[i] == NULL){
            fprintf(stderr, "Malloc failed on line %d\n", line);
            exit(-1);
        }
    }
}

//---------------------------------------------------------------------------------------Audit
void print_matrix(double* mat, int rows, int cols, char* msg){
    fprintf(stdout, "%s", msg);
    for (int i=0; i < rows; i++) {
        for (int j=0; j < cols; j++) {
            fprintf(stdout, "\t%20.19g", mat[i*cols+j]);
        }
        fprintf(stdout, "\n");
    }
}

void print_csr(CSR* csr){
    fprintf(stdout, "CSR:\n");
    fprintf(stdout, "\tM: %d, N: %d, NZ: %d\n", csr->M, csr->N, csr->NZ);
    fprintf(stdout, "\tRow pointers: [");
    for (int i=0; i<csr->M-1; i++) {
        fprintf(stdout, "%d, ", csr->IRP[i]);
    }
    fprintf(stdout, "%d]\n", csr->IRP[csr->M-1]);

    fprintf(stdout, "Value (Column):\n");
    for (int i=0; i<csr->NZ-1; i++) {
        fprintf(stdout, "\t%20.19g (%d)", csr->AS[i], csr->JA[i]);
        if (i%3 == 2) {
            fprintf(stdout, "\n");
        }
    }
    fprintf(stdout, "\t%20.19g (%d)\n", csr->AS[csr->NZ-1], csr->JA[csr->NZ-1]);
}

void print_ell(ELL* ell) {
    fprintf(stdout, "ELL:\n");
    fprintf(stdout, "\tM: %d, N: %d, MAXNZ: %d\n", ell->M, ell->N, ell->MAXNZ);
    fprintf(stdout, "Value (Column):\n\t");
    for (int i = 0; i < ell->M; i++) {
        for (int j = 0; j < ell->MAXNZ; j++) {
            fprintf(stdout, "%20.19g (%d) ", ell->AS[i*ell->MAXNZ + j], ell->JA[i*ell->MAXNZ + j]);
        }
        fprintf(stdout, "\n\t");
    }
}