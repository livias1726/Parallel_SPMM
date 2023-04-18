#include "omp_utils.h"

void spmm_csr(CSR*, const int*, int, const Type*, int, Type*);
void csr_nz_balancing(int, int, const int*, int, int*);
void csr_init_struct(Type* y, int* thread_rows, int num_threads, int k);
