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
void serial_product_csr(CSR* mat, const double* x, int k, double* y){
    int i, j, z, rows = mat->M, *irp = mat->IRP, *ja = mat->JA;
    double *t, *as = mat->AS;

    for (i = 0; i < rows; i++) {
        t = &y[i*k];

        for (j = irp[i]; j < irp[i+1]; j++) {
            for (z = 0; z < k; z++) {
                t[z] += as[j]*(x[ja[j]*k+z]);
            }
        }
    }
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
void serial_product_ell(ELL mat, const double* x, int k, double* y){
    int i, j, z, maxnz = mat.MAXNZ;
    double t, val;

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
}