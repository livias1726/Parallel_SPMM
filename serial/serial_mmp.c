#include "../utils/utils.h"
#include <string.h>

bool ELLPACK = 0;

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

    MM_typecode t;
    FILE *f;
    CSR* csr;
    ELL* ell;

    // check the correct use of the program
    if (argc < 3){
        fprintf(stderr, "Usage: %s [mm-filename] [storage-format]\n", argv[0]);
        exit(-1);
    } else if ((f = fopen(argv[1], "r")) == NULL) { //check the correct opening of the matrix file
        exit(-1);
    }

    // process the first line of file and identify the matrix type
    if (mm_read_banner(f, &t) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(-1);
    }

    check_mat_type(t);

    // convert to wanted storage format
    if (strcmp(argv[2], "ellpack") == 0) {
        ELLPACK = 1;
        ell = (ELL*) malloc(sizeof(ELL));
        error_handler(ell);

        read_mm_ell(f, &ell, t);
        fclose(f);
    } else {
        read_mm_csr(f, &csr, t);
        fclose(f);
    }

    // write matrix to stdout
    /*
    mm_write_banner(stdout, t);
    mm_write_mtx_crd_size(stdout, M, N, NZ);
    for (i=0; i < NZ; i++) {
        fprintf(stdout, "%d %d %20.19g\n", I[i] + 1, J[i] + 1, val[i]);
    }
    */

    //TODO: populate multivector
    //TODO: perform product
    //TODO: store output

    return 0;
}


