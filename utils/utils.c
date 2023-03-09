#include "utils.h"

void process_mm(MM_typecode* t, FILE *f){
    // process the first line of file and identify the matrix type
    if (mm_read_banner(f, t) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(-1);
    }

    // check matrix type support
    if ((!mm_is_real(*t) || !mm_is_pattern(*t)) && !mm_is_sparse(*t)) {
        printf("This application does not support Market Market type: [%s]\n", mm_typecode_to_str(*t));
        exit(-1);
    }
}

void populate_multivector(Type* vec, int rows, int cols) {
    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            vec[i*cols+j] = ((Type)rand()/RAND_MAX);
        }
    }
}

void alloc_struct(Type** vec, int rows, int cols) {
    *vec = (Type*) calloc(rows*cols, sizeof(Type));
    malloc_handler(1, (void*[]){*vec});
}

void malloc_handler(int size, void **p) {
    for(int i=0; i<size; i++){
        if(p[i] == NULL){
            fprintf(stderr, "Malloc failed.\n");
            exit(-1);
        }
    }
}

void clean_up(int size, void **p){
    for(int i=0; i<size; i++){
        free(p[i]);
    }
}

/**
 * Computes the absolute and relative error of the parallel product
 * computation (wrt the serial one) using the infinity norm.
 * */
void get_errors(int rows, int cols, Type* seq, Type* par, double* abs, double* rel){
    int i, j, idx;
    double max_diff = 0.0, tmp_diff, tmp_norm, norm = 0.0;

    for (i = 0; i < rows; i++) {
        tmp_diff = 0.0;
        tmp_norm = 0.0;
        idx = i*cols;

        for (j = 0; j < cols; j++) {
            tmp_diff += fabs(par[idx + j] - seq[idx + j]);
            tmp_norm += fabs(seq[idx + j]);
        }

        if (tmp_diff > max_diff) max_diff = tmp_diff;
        if (tmp_norm > norm) norm = tmp_norm;
    }

    *abs = max_diff;
    *rel = max_diff/norm;
}