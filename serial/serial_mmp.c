#include <stdlib.h>
#include "../utils/utils.h"

#define MIN -10
#define MAX 10

void alloc_structs(float** A, float** B, int dims) {
   *A = (float*)malloc(dims*dims*sizeof(float));
   *B = (float*)malloc(dims*dims*sizeof(float));
}

void populate_matrix(float* mat, int dims) {

   int i, j;
   for (i=0; i<dims; i++){
      for (j=0; j<dims; j++){
         mat[i*dims+j] = ((float)rand()/RAND_MAX)*(MAX - MIN) + MIN;
      }
   }
}

float* product_kji(float* A, float* B, float* C, int dims, struct timeval *t1, struct timeval *t2) {
   float* D = (float*)malloc(dims*dims*sizeof(float));
   if(D == NULL){
      fprintf(stderr, "Malloc failed.");
      exit(-1);
   }

   int i,j,k;
   gettimeofday(t1, NULL);
   for (k=0; k<dims; k++) {
      for (i=0; i<dims; i++){
         for(j=0; j<dims; j++){
            D[i*dims+j] = A[i*dims+k]*B[k*dims+j]+C[i*dims+j];
         }
      }
   }
   gettimeofday(t2, NULL);
   return D;
}

int main(int argc, char** argv) {

    float* A = NULL;
    float* B = NULL;
    float* C = NULL;

    struct timeval t1;
    struct timeval t2;

    int dims;

    alloc_structs(&A, &B, dims);
    populate_matrix(A, dims);
    populate_matrix(B, dims);
    if(A == NULL || B == NULL || C == NULL){
        fprintf(stderr, "Malloc failed.");
        exit(-1);
    }

    C = product_kji(A, B, C, dims, &t1, &t2);

    get_mflops(t2.tv_sec - t1.tv_sec, dims);

    free(A);
    free(B);
    free(C);

    return 0;
} 