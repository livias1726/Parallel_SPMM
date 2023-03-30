#include <omp.h>
#include "../../utils/headers/utils.h"

int* ell_nz_balancing(int ts, ELL* ell, int*, int*);
int* csr_nz_balancing(int ts, int tot_nz, const int* irp, int tot_rows);
void process_arguments(int argc, char** argv, FILE **f, int* k, int*);