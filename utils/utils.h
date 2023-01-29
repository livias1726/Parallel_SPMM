#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "../lib/mmio.h"

#ifndef SCPA_PROJECT_UTILS_H
#define SCPA_PROJECT_UTILS_H

#endif //SCPA_PROJECT_UTILS_H

/**
 * Non-zero element structure:
 * used to cope with column ordering
 * */
typedef struct elem {
    int i;
    int j;
    double val;
} Elem;

/**
 * Matrix structures
 * */
/**
 * CSR
 *
 * @param M number of rows
 * @param N number of cols
 * @param IRP array of row pointers: indexes of the values in AS that start a new row
 * @param JA array of non-zeros col indices
 * @param AS array of non-zero values
 * */
typedef struct matrix_csr {
    int M;
    int N;
    int* IRP;
    int* JA;
    double* AS;
} CSR;

/**
 * ELL
 *
 * @param M number of rows
 * @param N number of cols
 * @param MAXNZ max number of non-zeros in a row
 * @param JA 2D array of non-zeros col indices
 * @param AS 2D array of non-zero values
 * */
typedef struct matrix_ell {
    int M;
    int N;
    int MAXNZ;
    int* JA;
    double* AS;
} ELL;

/**
 * Functions signatures
 * */
void check_mat_type(MM_typecode);
void error_handler(void *p);
void read_mm_csr(FILE* f, CSR** mat, MM_typecode t);
void read_mm_ell(FILE* f, ELL** mat, MM_typecode t);
void get_mflops(time_t, int);