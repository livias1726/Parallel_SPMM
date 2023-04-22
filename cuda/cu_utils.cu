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

unsigned int alloc_cuda_spmm(Type **d_x, Type **d_y, const Type *x, int m, int n, int k){

    int size_partial = k * sizeof(Type);
    unsigned size_x = n * size_partial;
    unsigned size_y = m * size_partial;

    checkCudaErrors(cudaMalloc((void**) d_x, size_x));
    checkCudaErrors(cudaMalloc((void**) d_y, size_y));

    checkCudaErrors(cudaMemcpy(*d_x, x, size_x, cudaMemcpyHostToDevice));
    
    return size_x + size_y;
}

