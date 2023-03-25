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
    int m = mat->M, *irp = mat->IRP, *ja = mat->JA;
    Type *as = mat->AS;

    int ry, rx, j, z;
    Type val;

    for (int i = 0; i < m; i++) {
        ry = i*k;

        for (j = irp[i]; j < irp[i+1]; j++) {
            val = as[j];
            rx = ja[j]*k;

            for (z = 0; z < k; z++) {
                y[ry + z] += val * x[rx + z];
            }
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
    int maxnz = mat->MAXNZ, m = mat->M, *ja = mat->JA;
    Type *as = mat->AS;

    int ra, rx, ry, j, z, idx;
    Type val;
    for (int i = 0; i < m; i++) {
        ra = i*maxnz;
        ry = i*k;

        for (j = 0; j < maxnz; j++) {
            idx = ra+j;
            val = as[idx];
            //if (val == 0) break; // cannot break product for some matrices still have zero values in the file

            rx = ja[idx]*k;
            for (z = 0; z < k; z++) {
                y[ry+z] += val * x[rx+z];
            }
        }
    }
}