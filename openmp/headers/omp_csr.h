#include "omp_utils.h"

void spmm_csr(CSR*, const int*, int, const Type*, int, Type*);
void csr_nz_balancing(int, int, const int*, int, int*);
