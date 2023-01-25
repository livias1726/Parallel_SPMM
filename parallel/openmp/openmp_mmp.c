#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUMTHREADS 4

void alloc_structs(float** mat, float** vec, float** res, int rows, int cols) {
   *mat = (float*)malloc(rows*cols*sizeof(float));
   *vec = (float*)malloc(cols*sizeof(float)); 
   *res = (float*)malloc(rows*sizeof(float)); 
}

void populate_matrix(float* mat, int rows, int cols) {
   float fl = 0;

   FILE *f = fopen("../txt/matrix.txt", "r");
   for(int i=0; i<rows; i++){
      for(int j=0; j<cols; j++){
         fscanf(f, "%f", &fl);
         mat[i*cols+j] = fl;
      }
   }
   fclose(f);
}

void populate_vector(float* vec, int cols) {
   float fl = 0;

   FILE *f = fopen("../txt/vector.txt", "r");
   for(int i=0; i<cols; i++){
      fscanf(f, "%f", &fl);
      vec[i] = fl;
   }
   fclose(f);
}

void get_dimensions(int* rows, int* cols){
   FILE *f = fopen("../txt/dimensions.txt", "r");
   fscanf(f, "%d", rows);
   fscanf(f, "%d", cols);
   fclose(f);
}

void save_result(float* res, int rows) {
   int i;

   FILE *f = fopen("txt/result_omp.txt", "w");
   for (i=0; i<rows; i++){
      fprintf(f, "%f ", res[i]);
   }
   fprintf(f, "\n");
   fclose(f);
} 

int main(int argc, char** argv) {

   float* mat = NULL;
   float* vec = NULL;
   float* res = NULL;

   int rows, cols;

   get_dimensions(&rows, &cols);

   alloc_structs(&mat, &vec, &res, rows, cols);
   if(mat == NULL || vec == NULL || res == NULL){
      fprintf(stderr, "Malloc failed.");
      exit(-1);
   }

   populate_matrix(mat, rows, cols);
   populate_vector(vec, cols);

   int i, j, tid;

   #pragma omp parallel for schedule(static, cols) shared(res) private(i)
   for (i=0; i<cols*rows; i++) {
      res[i/cols] += mat[i]*vec[i%cols];
   }

   /*
   #pragma omp parallel for schedule(dynamic, cols) shared(mat, res, rows, cols) private(i, j, vec, tid)
   for (i=0; i<rows; i++) {
      res[i] = 0.0;
      for (j=0; j<cols; j++){
         res[i] += mat[i*cols+j]*vec[j];
      }
   }
   */

   save_result(res, cols);

   free(mat);
   free(vec);
   free(res);

   return 0;
} 