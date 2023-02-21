#include "utils.h"

/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param x multivector Nxk stored as 1D array
 * @param y receives product results Mxk stored as 1D array
 * @param t1 pointer to first timeval structure
 * @param t2 pointer to second timeval structure
 * */
void serial_product_csr(CSR mat, const double* x, int k, double* y, struct timespec *t1, struct timespec *t2){
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
void serial_product_ell(ELL mat, const double* x, int k, double* y, struct timespec *t1, struct timespec *t2){
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