#include <omp.h>
#include <immintrin.h>
#include "../../utils/headers/utils.h"

#define PD_STRIDE 8
#define PS_STRIDE 16
#define _MM16_1 _mm512_set1_epi32(1)    // 16 32-bit integers set to 1
#define _MM8_1 _mm256_set1_epi32(1)     // 8 32-bit integers set to 1
#define _MM8(i) (_mm256_set1_epi32(i))

void process_arguments(int argc, char** argv, FILE **f, int* k, int*);