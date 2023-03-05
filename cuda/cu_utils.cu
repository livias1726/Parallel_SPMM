#include "cu_utils.cuh"

void process_arguments(int argc, char** argv, FILE **f, int* k){
    if (argc < 3){
        fprintf(stderr, "Usage: %s [mm-filename] [k value]\n", argv[0]);
        exit(-1);
    }

    // create file path
    char path[PATH_MAX] = "resources/files/";
    strcat(path, argv[1]);

    //check the correct opening of the matrix file
    *f = fopen(path, "r");
    if (*f == NULL) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        exit(-1);
    }

    // get k value and desired storage format
    *k = (int)strtol(argv[2], NULL, 10);
}

/**
 * CPU code that calculates the number of rows of a CSR matrix that can fit into LDS entries of size BD.
 * Computes the number of rows to give each block (s.t. # NZ <= BD) and the total number of blocks to cover all rows.
 *
 * Implementation of Algorithm 2 of
 * 'Greathouse, Daga - Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format'
 * where BD is the fixed local size of the scratchpad memory
 *
 * @param rows total number of rows in A
 * @param irp row delimiters of CSR format
 * @param blocks output array of row blocks
 * */
int csr_adaptive_blocks(int rows, int* irp, int* blocks){

    int nz = 0, last = 0, ctr = 1;
    blocks[0] = 0;

    for (int i = 1; i < rows; i++) {
        nz += irp[i] - irp[i-1]; // count the sum of non-zeros in the considered rows

        /*
        if (nz == BD) { // fills up the local size
            last = i;
            blocks[ctr++] = i;
            nz = 0;
        } else if (nz > BD) {
            if (i - last > 1) { // the extra row will not fit
                blocks[ctr++] = i - 1;
                i--;
            } else if (i - last == 1) { // this row is too large
                blocks[ctr++] = i;
            }

            last = i;
            nz = 0;
        }
         */

        if (nz < BD) continue;

        // more than 1 row --> not enough space: decrease number of rows for the block
        if ((nz > BD) && (i - last > 1)) i--;
        // else: exactly 1 row --> too large: there are more non-zeros in a row than threads in a block

        last = i;
        blocks[ctr++] = i;
        nz = 0;
    }

    blocks[ctr++] = rows;
    return ctr;
}

void allocCudaCsr(CSR* csr, int **d_irp, int **d_ja, double **d_as){
    int m = csr->M, nz = csr->NZ;
    int *irp = csr->IRP, *ja = csr->JA;
    double *as = csr->AS;

    checkCudaErrors(cudaMalloc((void**) d_irp, (m+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) d_ja, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) d_as, nz*sizeof(double)));

    checkCudaErrors(cudaMemcpy(*d_irp, irp, (m+1)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_ja, ja, nz*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_as, as, nz*sizeof(double), cudaMemcpyHostToDevice));
}

void allocCudaSpmm(double **d_x, double **d_y, const double *x, int m, int n, int k){

    checkCudaErrors(cudaMalloc((void**) d_x, n * k * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) d_y, m * k * sizeof(double)));

    checkCudaErrors(cudaMemcpy(*d_x, x,  n * k * sizeof(double), cudaMemcpyHostToDevice));
}