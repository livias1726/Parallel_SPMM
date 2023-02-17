#include "utils.h"

//---------------------------------------------------------------------------------------Initialization
//TODO: see if first half of main can be merged for all modules

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
Elem** read_mm(FILE* f, int* m, int* n, int* nz, const MM_typecode t){ //TODO: check necessity to step through elems
    int r, c, i, onz;

    // process the matrix size information
    if (mm_read_mtx_crd_size(f, m, n, nz) != 0) { exit(-1); }
    onz = *nz;

    // array of lists of Elem: 1 per row
    Elem** elems = (Elem**) malloc((*m)* sizeof(Elem*));
    malloc_handler(1, (void* []) {elems}, 49);
    for (i = 0; i < *m; i++) { elems[i] = NULL; }

    // scan matrix
    for (i = 0; i < onz; i++){
        Elem* elem = (Elem*)malloc(sizeof(Elem));
        malloc_handler(1, (void* []) {elem}, 55);

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
            malloc_handler(1, (void* []) {elem}, 70);

            elem_s->val = elem->val;
            elem_s->j = r;
            elem_s->next = NULL;
            insert_in_row(elems, elem_s, c);

            *nz += 1;
        }
    }

    return elems;
}

CSR* alloc_csr(int m, int n, int nz){
    // alloc memory
    CSR* mat = (CSR*) malloc(sizeof(CSR));
    malloc_handler(1, (void* []) {mat}, 90);

    mat->IRP = (int*)malloc(m*sizeof(int));
    mat->JA = (int*)malloc(nz*sizeof(int));
    mat->AS = (double*)malloc(nz*sizeof(double));
    malloc_handler(3, (void* []) {mat->IRP, mat->JA, mat->AS}, 95);

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
    malloc_handler(1, (void* []) {mat}, 155);
    // calloc is used to avoid the addition of padding in a loop
    // 2D arrays are treated as 1D arrays
    mat->JA = calloc(m*(*maxnz), sizeof(int));
    mat->AS = (double*)calloc(m*(*maxnz), sizeof(double));
    malloc_handler(2, (void* []) {mat->JA, mat->AS}, 117);

    // populate ELLPACK format
    mat->M = m;
    mat->N = n;
    mat->NZ = nz;
    mat->MAXNZ = *maxnz;

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

    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            vec[i*cols+j] = ((double)rand()/RAND_MAX);
        }
    }
}

void read_multivector(double* vec, int rows, int cols) {

    int i,j;
    double fl = 0;

    FILE *f = fopen("matrix.txt", "r");
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            fscanf(f, "%lf", &fl);
            vec[i*cols+j] = fl;
        }
    }
    fclose(f);
}

void save_result(double* y, int rows, int cols) {
    int i, j;

    FILE *f = fopen("result.txt", "w");
    for (i=0; i<rows; i++){
        for (j=0; j<cols; j++) {
            fprintf(f, "%lf ", y[i*cols+j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void alloc_struct(double** vec, int rows, int cols) {
    *vec = (double*) malloc(rows*cols* sizeof(double));
    malloc_handler(1, (void*[]){*vec}, 147);
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