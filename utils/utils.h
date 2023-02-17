#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include "../lib/mmio.h"

#ifndef SCPA_PROJECT_UTILS_H
#define SCPA_PROJECT_UTILS_H

#endif //SCPA_PROJECT_UTILS_H

#define PATH_MAX 512
#define NAME_MAX 256
#define IO_MAX 1024

/**
 * Elem:
 *      Node of elements lists used to temporarily store the non-zeros read from files
 *
 * @param j column index
 * @param val value
 * @param nz (meta) used in the head node to count the total number of non-zeros in the row
 * @param next pointer to the next element in the row/list
 * */
typedef struct elem {
    int j;
    double val;
    int nz;
    struct elem* next;
} Elem;

//--------------------------------------------------Matrix structures
/**
 * CSR
 *
 * @param M number of rows
 * @param N number of cols
 * @param NZ number of non-zeros (necessary for mflops computation)
 * @param IRP array of row pointers: indexes of the values in AS that start a new row
 * @param JA array of non-zeros col indices
 * @param AS array of non-zero values
 * */
typedef struct csr {
    int M;
    int N;
    int NZ;
    int* IRP;
    int* JA;
    double* AS;
} CSR;

/**
 * ELL
 *
 * @param M number of rows
 * @param N number of cols
 * @param NZ number of non-zeros (necessary for mflops computation)
 * @param MAXNZ max number of non-zeros in a row
 * @param JA 2D array of non-zeros col indices
 * @param AS 2D array of non-zero values
 * */
typedef struct ell {
    int M;
    int N;
    int NZ;
    int MAXNZ;
    int* JA;
    double* AS;
} ELL;

//-------------------------------------------------Functions signatures
Elem** read_mm(FILE* f, int* m, int* n, int* nz, const MM_typecode t);
CSR* alloc_csr(int m, int n, int nz);
ELL* alloc_ell(Elem** elems, int m, int n, int nz, int* maxnz);

void check_mat_type(MM_typecode);
void malloc_handler(int, void**, int);
void populate_multivector(double*, int, int);
void save_result(double*, int, int);
void alloc_struct(double**, int, int);

void print_matrix(double*, int, int, char*);
void print_csr(CSR*);
void print_ell(ELL*);