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

int csr_adaptive_blocks(int rows, int* irp, int* blocks){

    int nz_row = 0, last = 0, ctr = 1;

    blocks[0] = 0;
    for (int i = 1; i < rows; i++) {
        nz_row += irp[i] - irp[i-1];

        if (nz_row == BD){ // row fills up block dim
            last = i;
            blocks[ctr++] = i;
            nz_row = 0;

        } else if (nz_row > BD){
            if (i - last > 1) { // not enough space
                blocks[ctr++] = i - 1;
                i--;
            } else if (i - last == 1) { // too large
                blocks[ctr++] = i;
            }

            last = i;
            nz_row = 0;
        }
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
    checkCudaErrors(cudaMemcpy(*d_as, as, nz*sizeof(int), cudaMemcpyHostToDevice));
}

void allocCudaSpmm(double **d_x, double **d_y, const double *x, int m, int n, int k){

    checkCudaErrors(cudaMalloc((void**) d_x, n * k * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) d_y, m * k * sizeof(double)));

    checkCudaErrors(cudaMemcpy(*d_x, x,  n * k * sizeof(double), cudaMemcpyHostToDevice));
}