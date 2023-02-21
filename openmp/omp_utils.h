#include <omp.h>
#include "../utils/utils.h"

#ifndef SCPA_PROJECT_OMP_UTILS_H
    #define SCPA_PROJECT_OMP_UTILS_H
#endif

CSR* read_mm_csr(FILE* f, MM_typecode t);
ELL* read_mm_ell(FILE* f, MM_typecode t);
int* nz_balancing(int ts, int tot_nz, const int* irp, int tot_rows);