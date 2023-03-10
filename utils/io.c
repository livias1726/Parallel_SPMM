#include "headers/utils.h"

void read_multivector(Type* vec, int rows, int cols) {
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

void save_result(Type* y, int rows, int cols) {
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

void print_matrix(Type* mat, int rows, int cols, char* msg){
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