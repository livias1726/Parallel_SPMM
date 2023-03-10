#include "headers/omp_utils.h"

//TODO:
// what if the single row has more NZs than the ones to assign to the single thread? Try to divide by blocks
// ordina le righe di ELL per numero di NZ ??
void sort_rows(ELL* ell, int *idxs){
    int idx = 0, prev_nz = 0, nz = 0, cols = ell->MAXNZ, rows = ell->M;
    double *as = ell->AS;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (as[i*cols + j] == 0) break;
            nz++;
        }

        if (nz > prev_nz) { // put nz before prev_nz
            idxs[idx++] = nz;
            idxs[idx++] = prev_nz;
        } else {
            idxs[idx++] = prev_nz;
            idxs[idx++] = nz;
        }

        nz = 0;
    }
}

void process_arguments(int argc, char** argv, FILE **f, int* k, int* num_threads){
    if (argc < 4){
        fprintf(stderr, "Usage: %s [mm-filename] [k value] [num-threads]\n", argv[0]);
        exit(-1);
    }

    // create file path
    char path[PATH_MAX] = "resources/files/";
    strcat(path, argv[1]);

    //check the correct opening of the matrix file
    *f = fopen(path, "r");
    if (*f == NULL) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        exit(-1);
    }

    // get k value and desired storage format
    *k = (int)strtol(argv[2], NULL, 10);

    // set number of threads
    *num_threads = (int)strtol(argv[3], NULL, 10);
}