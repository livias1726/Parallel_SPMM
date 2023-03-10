#include "headers/utils.h"

/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void serial_product_csr(CSR* mat, const Type* x, int k, Type* y){
    int i, j, z, rows = mat->M, *irp = mat->IRP, *ja = mat->JA;
    Type t, *as = mat->AS;

    for (i = 0; i < rows; i++) {
        for (z = 0; z < k; z++) {
            t = 0.0;
            for (j = irp[i]; j < irp[i+1]; j++) {
                t += as[j]*(x[ja[j]*k+z]);
            }
            y[i*k + z] = t;
        }
    }
}

/**
 * Computes the product with A stored in a ELL format
 *
 * @param mat matrix in ellpack format
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void serial_product_ell(ELL* mat, const Type* x, int k, Type* y){
    int i, j, z, maxnz = mat->MAXNZ, rows = mat->M, *ja = mat->JA;
    Type t, val, *as = mat->AS;

    for (i = 0; i < rows; i++) {
        for (z = 0; z < k; z++) {
            t = 0.0;
            for (j = 0; j < maxnz; j++) { // TODO: try to use this as external loop
                val = as[i*maxnz+j];
                if (val == 0) break; // if padding is reached break loop
                t += val * x[ja[i*maxnz+j]*k+z];
            }
            y[i*k+z] = t;
        }
    }
}