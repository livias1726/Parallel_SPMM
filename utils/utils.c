#include "headers/utils.h"

void process_mm(MM_typecode* t, FILE *f){
    // process the first line of file and identify the matrix double
    if (mm_read_banner(f, t) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(-1);
    }

    // check matrix double support
    if ((!mm_is_real(*t) || !mm_is_pattern(*t)) && !mm_is_sparse(*t)) {
        printf("This application does not support Market Market double: [%s]\n", mm_typecode_to_str(*t));
        exit(-1);
    }
}

void populate_multivector(double* vec, int rows, int cols) {
    int i,j;
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            vec[i*cols+j] = ((double)rand()/RAND_MAX);
        }
    }
}

void alloc_struct(double** vec, int rows, int cols) {
    *vec = (double*) calloc(rows*cols, sizeof(double));
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
void get_errors(int elems, double* seq, double* par, double* abs, double* rel){

    double max_abs, diff_abs, diff = 0.0, rel_diff = 0.0;

    for (int i = 0; i < elems; i++) {
        max_abs = MAX(fabs(seq[i]), fabs(par[i]));

        if (max_abs == 0.0) max_abs = 1.0;

        diff_abs = fabs(seq[i]-par[i]);
        rel_diff = MAX(rel_diff, diff_abs/max_abs);

        diff = MAX(diff, diff_abs);
    }

    *abs = diff;
    *rel = rel_diff;
}

 /*
void get_errors(int elems, double* seq, double* par, double* abs, double* rel){

    double max_diff = 0.0, max_norm = 0.0;

    for (int i = 0; i < elems; i++) {
        max_diff = MAX(max_diff, fabs(par[i] - seq[i]));
        max_norm = MAX(max_norm, fabs(seq[i]));
    }

    *abs = max_diff;
    *rel = max_diff/max_norm;
}
  */

void tokenize_output (char* output, double* gs, double* gp, double* ae, double* re) {
    char* tok = strtok(output, " ");
    char* tokens[4];
    int count = -1;

    while (tok != NULL && count < 4) {
        count++;
        tokens[count] = tok;
        tok = strtok(NULL, " ");
    }

    if (count == 3) {
        *gs = strtod(tokens[0], NULL);
        *gp = strtod(tokens[1], NULL);
        *ae = strtod(tokens[2], NULL);
        *re = strtod(tokens[3], NULL);
    }
}
