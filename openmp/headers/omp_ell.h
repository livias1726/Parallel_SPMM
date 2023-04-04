#include "omp_utils.h"

void spmm_ell(ELL*, int, Type*, int k, Type* y);
void ell_nz_balancing(ELL*, int,  int*, int*);