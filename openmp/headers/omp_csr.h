#include "omp_utils.h"

#define PD_STRIDE 8
#define PS_STRIDE 16
#define _MM16_1 _mm512_set1_epi32(1)    // 16 32-bit integers set to 1
#define _MM8_1 _mm256_set1_epi32(1)     // 8 32-bit integers set to 1

void spmm_csr(CSR*, const int*, int, const Type*, int, Type*);
void csr_nz_balancing(int, int, const int*, int, int*);
