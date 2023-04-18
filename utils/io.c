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
    //fprintf(stdout, "\tRow pointers: [");

    int i;
    /*
    for (i = 0; i < m; i++) {
        fprintf(stdout, "%d, ", irp[i]);
    }
    fprintf(stdout, "%d]\n", irp[m]);
     */

    fprintf(stdout, "Value (Column):\n");
    for (i = 0; i < 10; i++) {
        fprintf(stdout, "\t%20.19g (%d)\n", as[i], ja[i]);
    }
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

// TODO ------------------------------------------- Added for testing -------------------------------------------------
void save_csr(CSR* csr, char* name){
    int i, j;

    // create file path
    name[strlen(name) - 4] = 0;

    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "resources/tmp/%s_csr.txt", name);

    int m = csr->M, n = csr->N, nz = csr->NZ;
    int *ja = csr->JA, *irp = csr->IRP;
    Type *as = csr->AS;

    FILE *f = fopen(path, "w");

    // first row for m, n, nz
    fprintf(f, "%d %d %d\n", m, n, nz);

    for (i = 0; i < m; i++){
        for (j = irp[i]; j < irp[i+1]; j++) {
            fprintf(f, "%d %d %20.19g\n", i, ja[j], as[j]);
        }
    }
    fclose(f);
}

CSR* read_csr(char* name){
    // create file path
    name[strlen(name) - 4] = 0;
    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "resources/tmp/%s_csr.txt", name);

    int m, n, nz;
    FILE *f = fopen(path, "r");
    fscanf(f, "%d %d %d\n", &m, &n, &nz);
    CSR* csr = alloc_csr(m, n, nz);

    int row_ctr = 0, i, row_prev = 0, row;
    csr->IRP[0] = 0;
    for(i = 0; i < nz; i++){
        fscanf(f, "%d %d %lg\n", &row, &csr->JA[i], &csr->AS[i]);

        if (row != row_prev) {
            row_prev = row;
            csr->IRP[++row_ctr] = i;
        }
    }
    fclose(f);

    return csr;
}

void save_ell(ELL* ell, char* name) {
    int i, j;

    // create file path
    name[strlen(name) - 4] = 0;

    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "resources/tmp/%s_ell.txt", name);

    int m = ell->M, n = ell->N, nz = ell->NZ, maxnz = ell->MAXNZ;
    int *ja = ell->JA;
    Type *as = ell->AS;

    FILE *f = fopen(path, "w");

    // first row for m, n, nz
    fprintf(f, "%d %d %d %d\n", m, n, nz, maxnz);

    int row;
    for (i = 0; i < m; i++){
        row = i * maxnz;
        for (j = 0; j < maxnz; j++) {
            fprintf(f, "%d %d %20.19g\n", i, ja[row + j], as[row + j]);
        }
    }
    fclose(f);
}

ELL* read_ell(char* name){
    // create file path
    name[strlen(name) - 4] = 0;
    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "resources/tmp/%s_ell.txt", name);

    int m, n, nz, maxnz;
    FILE *f = fopen(path, "r");
    fscanf(f, "%d %d %d %d\n", &m, &n, &nz, &maxnz);
    ELL* ell = alloc_ell(m, n, nz, maxnz);

    int row;
    for (int i = 0; i < m * maxnz; i++){
        fscanf(f, "%d %d %lg\n", &row, &ell->JA[i], &ell->AS[i]);
    }
    fclose(f);

    return ell;
}