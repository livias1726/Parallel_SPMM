#include "headers/cu_utils.cuh"

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

void alloc_cuda_csr(CSR* csr, int **d_irp, int **d_ja, Type **d_as){
    int m = csr->M;
    int nz = csr->NZ;
    int size_irp = (m+1)*sizeof(int);
    int size_ja = nz*sizeof(int);
    int size_as = nz*sizeof(Type);

    int *irp = csr->IRP, *ja = csr->JA;
    Type *as = csr->AS;

    checkCudaErrors(cudaMalloc((void**) d_irp, size_irp));
    checkCudaErrors(cudaMalloc((void**) d_ja, size_ja));
    checkCudaErrors(cudaMalloc((void**) d_as, size_as));

    checkCudaErrors(cudaMemcpy(*d_irp, irp, size_irp, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_ja, ja, size_ja, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(*d_as, as, size_as, cudaMemcpyHostToDevice));
}

void alloc_cuda_spmm(Type **d_x, Type **d_y, const Type *x, int m, int n, int k){

    int size_partial = k * sizeof(Type);
    int size_x = n * size_partial;

    checkCudaErrors(cudaMalloc((void**) d_x, size_x));
    checkCudaErrors(cudaMemcpy(*d_x, x,  size_x, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) d_y, (m*size_partial)));
}



