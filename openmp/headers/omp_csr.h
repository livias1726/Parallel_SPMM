#include "omp_utils.h"

#define PD_STRIDE 8
#define PS_STRIDE 16

#define _MM16_1 _mm512_set1_epi32(1)
#define _MM8_1 _mm256_set1_epi32(1)
#define _MM4_1 _mm_set1_epi32(1)

void spmm_csr(CSR*, const int*, int, const Type*, int, Type*);
void csr_nz_balancing(int, int, const int*, int, int*);
