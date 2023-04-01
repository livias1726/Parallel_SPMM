#include <omp.h>
#include <immintrin.h>
#include "../../utils/headers/utils.h"

#define L1_CACHE 32*1024

void process_arguments(int argc, char** argv, FILE **f, int* k, int*);