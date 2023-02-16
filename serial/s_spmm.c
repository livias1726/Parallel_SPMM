#include "../utils/utils.h"

/**
 * serial_mmp performs a SERIAL matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 * */

//---------------------------------------------------------------------------------------------------Pre-processing
/**
 * Read the matrix into a CSR struct representing the matrix in CSR storage format
 *
 * @param f file descriptor
 * @param t matrix type code
 * */
CSR* read_mm_csr(FILE* f, MM_typecode t){
    int i, m, n, nz, elem_count = 0;
    Elem *curr, *prev;

    // read matrix from file
    Elem** elems = read_mm(f, &m, &n, &nz, t);
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

//---------------------------------------------------------------------------------------------------Product

/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param x multivector Nxk stored as 1D array
 * @param y receives product results Mxk stored as 1D array
 * @param t1 pointer to first timeval structure
 * @param t2 pointer to second timeval structure
 * */
void product_csr(CSR mat, const double* x, int k, double* y, struct timespec *t1, struct timespec *t2){
    int i, j, z, limit, rows = mat.M;
    double t;

    clock_gettime(CLOCK_MONOTONIC, t1);
    for (z = 0; z < k; z++) {
        for (i = 0; i < rows; i++) {
            t = 0.0;

            limit = (i != rows-1) ? mat.IRP[i+1] : mat.NZ;
            for (j = mat.IRP[i]; j < limit; j++) {
                t += mat.AS[j]*(x[mat.JA[j]*k+z]);
            }
            y[i*k+z] = t;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, t2);
}

/**
 * Computes the product with A stored in a ELL format
 *
 * @param mat matrix in ellpack format
 * @param x multivector Nxk stored as 1D array
 * @param y receives product results Mxk stored as 1D array
 * @param t1 pointer to first timeval structure
 * @param t2 pointer to second timeval structure
 * */
void product_ell(ELL mat, const double* x, int k, double* y, struct timespec *t1, struct timespec *t2){
    int i, j, z, maxnz = mat.MAXNZ;
    double t, val;

    clock_gettime(CLOCK_MONOTONIC, t1);
    for (z = 0; z < k; z++) {
        for (i = 0; i < mat.M; i++) {
            t = 0.0;

            for (j = 0; j < maxnz; j++) {
                val = mat.AS[i*maxnz+j];
                if (val == 0) { // if padding is reached break loop
                    break;
                }
                t += val*x[mat.JA[i*maxnz+j]*k+z];
            }

            y[i*k+z] = t;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, t2);
}

//-----------------------------------------------------------------------------------------Main
int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR* csr = NULL;
    ELL* ell = NULL;
    double *x, *y;
    int k, m, n, nz;
    struct timespec t1, t2;
    bool ellpack;

    // check the correct use of the program
    if (argc < 4){
        fprintf(stderr, "Usage: %s [mm-filename] [storage-format] [k value] \n", argv[0]);
        exit(-1);
    }

    // create file path
#ifdef PERFORMANCE
    char path[PATH_MAX] = "resources/files/";
#else
    char path[PATH_MAX] = "../resources/files/";
#endif
    strcat(path, argv[1]);

    //check the correct opening of the matrix file
    f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        exit(-1);
    }

    // get k value and desired storage format
    ellpack = (strcmp(argv[2], "ellpack") == 0) ? true : false;
    k = (int)strtol(argv[3], NULL, 10);

    // process the first line of file and identify the matrix type
    if (mm_read_banner(f, &t) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(-1);
    }

    // check matrix type support
    check_mat_type(t);

    // convert to wanted storage format
    if (ellpack) {
        ell = read_mm_ell(f, t);
        m = ell->M;
        n = ell->N;
        nz = ell->NZ;
#ifdef AUDIT
        print_ell(ell);
#endif
    } else {
        csr = read_mm_csr(f, t);
        m = csr->M;
        n = csr->N;
        nz = csr->NZ;
#ifdef AUDIT
        print_csr(csr);
#endif
    }

    fclose(f);

    alloc_struct(&x, n, k);
    alloc_struct(&y, m ,k);

    populate_multivector(x, n, k);

#ifdef AUDIT
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

    // compute the product
    if (ellpack) {
        product_ell(*ell, x, k, y, &t1, &t2);
        free(ell);
    } else {
        product_csr(*csr, x, k, y, &t1, &t2);
        free(csr);
    }

    free(x);

#ifdef AUDIT
    // print results
    print_matrix(y, m, k, "\nResult:\n");
#endif

#ifdef PERFORMANCE
    fprintf(stdout, "%ld.%.9ld %ld.%.9ld %d", t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec, nz);
#endif

    //fprintf(stdout, "\n%ld.%.9ld %ld.%.9ld\n", t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
    return 0;
}