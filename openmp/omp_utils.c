#include "headers/omp_utils.h"

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

void clear_cache(Type *cooler, int threads){
    #pragma omp parallel for num_threads(threads) shared(cooler) default(none)
    for (int i = 0; i < L3/4; ++i) {
        cooler[i] = 0;
    }
}