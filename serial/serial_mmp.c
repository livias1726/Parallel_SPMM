#include "../utils/utils.h"
#include <string.h>

/**
 * serial_mmp performs a SERIAL matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 * */

bool ELLPACK = 0;

//-------------------------------------------------------Product

/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param x multivector Nxk stored as 1D array
 * @param y receives product results Mxk stored as 1D array
 * @param t1 pointer to first timeval structure
 * @param t2 pointer to second timeval structure
 * */
void product_csr(CSR mat, const double* x, int k, double** y, struct timeval *t1, struct timeval *t2){
    int i, j, z, rows = mat.M;
    double t;

    *y = (double*)malloc(rows*k*sizeof(double));
    error_handler(*y);

    gettimeofday(t1, NULL);
    for (z = 0; z < k; z++) {
        for (i = 0; i < rows; i++) {
            t = 0.0;

            for (j = mat.IRP[i]; j < mat.IRP[i+1]-1; j++) {
                t += mat.AS[j]*x[mat.JA[j]*k+z];
            }

            *y[i*k+z] = t;
        }
    }
    gettimeofday(t2, NULL);
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
void product_ell(ELL mat, const double* x, int k, double** y, struct timeval *t1, struct timeval *t2){
    int i, j, z, m = mat.M, maxnz = mat.MAXNZ;
    double t;

    *y = (double*)malloc(m*k*sizeof(double));
    error_handler(*y);

    gettimeofday(t1, NULL);
    for (z = 0; z < k; z++) {
        for (i = 0; i < m; i++) {
            t = 0.0;

            for (j = 0; j < maxnz; j++) {
                t += mat.AS[i*maxnz+j]*x[mat.JA[i*maxnz+j]*k+z];
            }

            *y[i*k+z] = t;
        }
    }
    gettimeofday(t2, NULL);
}

//---------------------------------------------------Utils
/**
 * Populates x with random doubles
 *
 * @param vec receives the multivector
 * @param rows number of rows of the multivector
 * @param cols number of cols of the multivector
 * */
void populate_vector(double** vec, int rows, int cols) {
    int i, j;

    *vec = (double*) malloc(rows*cols* sizeof(double));
    error_handler(*vec);

    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            (*vec)[i*rows+j] = ((double)rand()/RAND_MAX);
        }
    }
}

//-----------------------------------------------------Main
int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR* csr;
    ELL* ell;
    double *x, *y;
    int k, m, n;
    struct timeval t1, t2;

    // check the correct use of the program
    if (argc < 4){
        fprintf(stderr, "Usage: %s [mm-filename] [k] [storage-format]\n", argv[0]);
        exit(-1);
    }

    // create file path
    char path[] = "C:\\Users\\oem\\OneDrive - Universita' degli Studi di Roma Tor Vergata\\Corsi\\Attivi\\SCPA\\Progetto\\SCPA_Project\\resources\\";
    strcat(path, argv[1]);

    //check the correct opening of the matrix file
    f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        exit(-1);
    }

    // get k value and desired storage format
    k = strtol(argv[2], NULL, 10);
    ELLPACK = (strcmp(argv[2], "ellpack") == 0) ? true : false;

    // process the first line of file and identify the matrix type
    if (mm_read_banner(f, &t) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(-1);
    }

    // check matrix type support
    check_mat_type(t);

    // convert to wanted storage format
    if (ELLPACK) {
        ell = read_mm_ell(f, t);
        m = ell->M;
        n = ell->N;
    } else {
        csr = read_mm_csr(f, t);
        m = csr->M;
        n = csr->N;
    }

    fclose(f);

    populate_vector(&x, n, k);

    // compute the product
    if (ELLPACK) {
        product_ell(*ell, x, k, &y, &t1, &t2);
    } else {
        product_csr(*csr, x, k, &y, &t1, &t2);
    }

    int dims[3] = {m, n, k};
    get_mflops(t2.tv_sec-t1.tv_sec, dims, 3);

    //TODO: store output
    /*write matrix to stdout
    mm_write_banner(stdout, t);
    mm_write_mtx_crd_size(stdout, M, N, NZ);
    for (i=0; i < NZ; i++) {
        fprintf(stdout, "%d %d %20.19g\n", I[i] + 1, J[i] + 1, val[i]);
    }
    */

    return 0;
}