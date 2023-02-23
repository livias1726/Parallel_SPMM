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

void populate_multivector(double* vec1, int rows, int cols) {
    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            vec1[i*cols+j] = ((double)rand()/RAND_MAX);
        }
    }
}

void alloc_struct(double** vec, int rows, int cols) {
    *vec = (double*) malloc(rows*cols* sizeof(double));
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