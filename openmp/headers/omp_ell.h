#include "omp_utils.h"

void spmm_ell(ELL* mat, int threads, double* x, int k, double* y);
int* ell_nz_balancing(int ts, ELL* ell, int* ordered_rows, int* rows_idx);