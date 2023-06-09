#include <omp.h>
#include <immintrin.h>
#include "../../utils/headers/utils.h"

#define PD_STRIDE 8
#define PS_STRIDE 16
#define MAX_NUM_THREADS 40
#define MAX_NUM_RUNS 10

#define MM16(i) _mm512_set1_epi32(i)   // 16 32-bit integers set to i
#define MM8(i) _mm256_set1_epi32(i)    // 8 32-bit integers set to 1

void process_arguments(int argc, char** argv, FILE **f, int* k, int*);
void clear_cache(Type *cooler, int threads);