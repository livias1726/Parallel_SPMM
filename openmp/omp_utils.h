#include <omp.h>
#include "../utils/utils.h"

#ifndef SCPA_PROJECT_OMP_UTILS_H
    #define SCPA_PROJECT_OMP_UTILS_H
#endif

int* nz_balancing(int ts, int tot_nz, const int* irp, int tot_rows);
void process_arguments(int argc, char** argv, FILE **f, int* k, int*);