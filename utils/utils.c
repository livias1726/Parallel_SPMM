#include "utils.h"

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

void initialize_array(Elem** arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = NULL;
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
    // array of lists of Elem: 1 per row
    initialize_array(elems, m);

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
 * @param mat structure to populate
 * @param nz pointer to the number of non-zeros (to be eventually updated)
 * @param t matrix type code
 * */
CSR* read_mm_csr(FILE* f, MM_typecode t){
    int ret, i, m, n, nz, elem_count = 0;
    Elem *curr, *prev;
    CSR* mat;

    // process the matrix size information
    ret = mm_read_mtx_crd_size(f, &m, &n, &nz);
    if (ret != 0) { exit(-1); }

    // read matrix from file
    Elem* elems[m];
    read_mm(f, elems, m, &nz, t);

    // alloc memory
    mat = (CSR*) malloc(sizeof(CSR));
    error_handler(mat);

    mat->IRP = (int*)malloc(m*sizeof(int));
    mat->JA = (int*)malloc(nz*sizeof(int));
    mat->AS = (double*)malloc(nz*sizeof(double));
    error_handler(mat->IRP);
    error_handler(mat->JA);
    error_handler(mat->AS);

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

    return mat;
}

ELL* read_mm_ell(FILE* f, MM_typecode t){
    int ret, i, m, n, nz, maxnz, count = 0;
    Elem *curr, *prev;
    ELL* mat;

    // process the matrix size information
    ret = mm_read_mtx_crd_size(f, &m, &n, &nz);
    if (ret != 0) { exit(-1); }

    // read matrix from file
    Elem* elems[m];
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
    error_handler(mat);
    // calloc is used to avoid the addition of padding in a loop
    // 2D arrays are treated as 1D arrays
    mat->JA = (int*)calloc((m-1)*(maxnz-1), sizeof(int));
    mat->AS = (double*)calloc((m-1)*(maxnz-1), sizeof(double));
    error_handler(mat->JA);
    error_handler(mat->AS);

    // populate ELLPACK format
    mat->M = m;
    mat->N = n;
    mat->MAXNZ = maxnz;

    // scan the array of lists: 1 per row
    for (i = 0; i < m; i++){
        curr = elems[i];

        // skip empty rows
        if (curr == NULL) { continue; }

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

    return mat;
}

void check_mat_type(MM_typecode t) {
    if ((!mm_is_real(t) || !mm_is_pattern(t)) && !mm_is_sparse(t)) {
        printf("This application does not support Market Market type: [%s]\n", mm_typecode_to_str(t));
        exit(-1);
    }
}

/**
 * Populates x with random doubles
 *
 * @param vec receives the multivector
 * @param rows number of rows of the multivector
 * @param cols number of cols of the multivector
 * */
void populate_multivector(double* vec, int rows, int cols) {
    int i, j;

    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            vec[i*cols+j] = ((double)rand()/RAND_MAX);
        }
    }
}

void alloc_struct(double** vec, int rows, int cols) {
    *vec = (double*) malloc(rows*cols* sizeof(double));
    error_handler(*vec);
}

/**
 * Computes MFLOPS of the product
 * */
void get_mflops(time_t v, const int* dims, int size){

    int num_ops = 2;

    for (int i = 0; i < size; i++) {
        num_ops *= dims[i];
    }

    fprintf(stdout, "MFLOPS: %f\n", num_ops/((double)v*pow(10,6)));
}

void error_handler(void *p) {
    if(p == NULL){
        fprintf(stderr, "Malloc failed");
        exit(-1);
    }
}

//------------------------------------------------Audit
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

void print_ell(ELL* ell){

}