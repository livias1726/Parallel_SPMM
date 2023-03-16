#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include "../../lib/mmio.h"

#ifndef SCPA_PROJECT_UTILS_H
    #define SCPA_PROJECT_UTILS_H
#endif

#define GET_GFLOPS(t1, t2, flop) ( flop / ((t2.tv_sec - t1.tv_sec) * 1.e9 + (t2.tv_nsec - t1.tv_nsec)) )
#define ROUND_UP(a,b) ((a+b-1)/b)
#define Type double

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
    Type val;
    int nz;
    struct elem* next;
} Elem;

//--------------------------------------------------Matrix structures
/**
 * CSR
 *
 * @param M number of rows
 * @param N number of cols
 * @param NZ number of non-zeros
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
    Type* AS;
} CSR;

/**
 * ELL
 *
 * @param M number of rows
 * @param N number of cols
 * @param NZ number of non-zeros
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
    Type* AS;
} ELL;

//-------------------------------------------------Functions signatures
// io
void read_multivector(Type*, int, int);
void save_result(Type*, int, int);
void print_matrix(Type*, int, int, char*);
void print_csr(CSR*);
void print_ell(ELL*);

// serial
void serial_product_csr(CSR*, const Type*, int, Type*);
void serial_product_ell(ELL*, const Type*, int, Type*);

// storage
Elem** read_mm(FILE*, int*, int*, int*, const MM_typecode);
CSR* read_mm_csr(Elem**, int, int, int);
ELL* read_mm_ell(Elem**, int, int, int);
CSR* alloc_csr(int, int, int);
ELL* alloc_ell(Elem**, int, int, int, int*);

// utils
void alloc_struct(Type**, int, int);
void process_mm(MM_typecode*, FILE*);
void malloc_handler(int, void**);
void clean_up(int, void**);
void populate_multivector(Type*, int, int);
void get_errors(int, int, Type*, Type*, double*, double*);









