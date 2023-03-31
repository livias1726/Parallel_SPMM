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

void save_result(char* name, Type* y, int rows, int cols) {
    int i, j;

    // create file path
    name[strlen(name) - 4] = 0;

    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "resources/results/%s_%d.txt", name, cols);

    FILE *f = fopen(path, "w");
    for (i=0; i<rows; i++){
        for (j=0; j<cols; j++) {
            fprintf(f, "%.16g ", y[i*cols+j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void print_matrix(Type* mat, int rows, int cols, char* msg){
    int i, j;
    fprintf(stdout, "%s", msg);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            fprintf(stdout, "\t%.16g", mat[i*cols+j]);
        }
        fprintf(stdout, "\n");
    }
}

void print_csr(CSR* csr){
    int m = csr->M, n = csr->N, nz = csr->NZ, *ja = csr->JA, *irp = csr->IRP;
    Type *as = csr->AS;

    fprintf(stdout, "CSR:\n");
    fprintf(stdout, "\tM: %d, N: %d, NZ: %d\n", m, n, nz);
    fprintf(stdout, "\tRow pointers: [");

    int i;
    for (i = 0; i < m; i++) {
        fprintf(stdout, "%d, ", irp[i]);
    }
    fprintf(stdout, "%d]\n", irp[m]);

    fprintf(stdout, "Value (Column):\n");
    for (i = 0; i < nz-1; i++) {
        fprintf(stdout, "\t%.16g (%d)", as[i], ja[i]);
        if (i%3 == 2) fprintf(stdout, "\n");
    }
    fprintf(stdout, "\t%.16g (%d)\n", as[nz-1], ja[nz-1]);
}

void print_ell(ELL* ell) {
    int m = ell->M, n = ell->N, maxnz = ell->MAXNZ, *ja = ell->JA;
    Type *as = ell->AS;

    fprintf(stdout, "ELL:\n");
    fprintf(stdout, "\tM: %d, N: %d, MAXNZ: %d\n", m, n, maxnz);
    fprintf(stdout, "Value (Column):\n\t");

    int row, idx;
    for (int i = 0; i < m; i++) {
        row = i * maxnz;
        for (int j = 0; j < maxnz; j++) {
            idx = row + j;
            fprintf(stdout, "%.16g (%d)\t\t", as[idx], ja[idx]);
        }
        fprintf(stdout, "\n\t");
    }
}