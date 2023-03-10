#include "omp_utils.h"

void spmm_csr(CSR *mat, const int* rows_load, int threads, double* x, int k, double* y);
int* csr_nz_balancing(int ts, int tot_nz, const int* irp, int tot_rows);
