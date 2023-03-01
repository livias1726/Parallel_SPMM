#include "omp_utils.h"

/**
 * Load balancing related to the amount of non-zeros given to each computational node.
 * The number of non-zeros is always rounded to be contained in a full row to maintain locality
 *
 * OPENMP:
 *      non-zeros are balanced on the number of threads that will operate the product
 * MPI:
 *      non-zeros are balanced on the number of processes that will operate the product
 *      inside every process - openmp threads will work on the given rows in parallel
 * */
int* nz_balancing(int ts, int tot_nz, const int* irp, int tot_rows){
    int i, j, r1, nz, start_row = 0, r2 = 0;

    int* rows_idx = (int*) malloc((ts+1) * sizeof(int));
    malloc_handler(1, (void*[]){rows_idx});

    for (i = 0; i < ts; i++) {
        rows_idx[i] = start_row; // add the idx of the start row

        if (i == ts-1) { // if last thread, get the remaining rows
            rows_idx[i+1] = tot_rows;
            break;
        }

        nz = ((i + 1) * tot_nz) / ts - (i * tot_nz) / ts; // compute the number of tot_nz to assign the i-th thread

        for (j = start_row; j < tot_rows; j++) {
            r2 += irp[j + 1] - irp[j]; // get number of nz in the considered rows

            if (r2 < nz) { // if the count of nz is still lower than the number of nz assigned to the thread
                r1 = r2; // save value
            } else {
                // get the number of rows that includes a number of nz closer to the one assigned
                start_row = ((r2 - nz) < (nz - r1)) ? j+1 : j;
                break;
            }
        }

        r2 = 0;
    }

    return rows_idx;
}

void process_arguments(int argc, char** argv, FILE **f, int* k, int* num_threads){
    if (argc < 4){
        fprintf(stderr, "Usage: %s [mm-filename] [k value] [num-threads]\n", argv[0]);
        exit(-1);
    }

    // create file path
    char path[PATH_MAX] = "../resources/files/";
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