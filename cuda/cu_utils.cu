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

/**
 * CPU code that calculates the number of rows of a CSR matrix that can fit into LDS entries of size BD.
 * Computes the number of rows to give each block (s.t. # NZ <= BD) and the total number of blocks to cover all rows.
 *
 * In addition, this returns the maximum number of NZs assigned to a block. Used in the computation of the
 * dynamic shared memory size.
 *
 * Inspired by Algorithm 2 of
 * 'Greathouse, Daga - Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format'
 * where BD is the fixed local size of the scratchpad memory.
 *
 * @param rows total number of rows in A
 * @param irp row delimiters of CSR format
 * @param blocks output array of row blocks
 * */
int get_csr_row_blocks(int rows, int* irp, int* blocks, int *max_nz){

    int nz = 0, max = 0, last_nz = 0, last_i = 0, ctr = 1;
    blocks[0] = 0;

    for (int i = 1; i < rows; i++) {
        nz += irp[i] - irp[i-1]; // count the sum of non-zeros in the considered rows

        if (nz == BDX) { // fills up the local size
            last_i = i;
            blocks[ctr++] = i;
            GET_MAX(max, nz)
            nz = 0;

        } else if (nz > BDX) {
            if (i - last_i > 1) { // more than 1 row --> not enough space: decrease number of rows for the block
                i--;
                GET_MAX(max, last_nz)
            } else { //last_i cannot be lower than i+1
                // exactly 1 row --> too large: there are more non-zeros in a row than threads in a block
                GET_MAX(max, nz)
            }

            last_i = i;
            blocks[ctr++] = i;
            nz = 0;
        }

        GET_MAX(max, nz)
        last_nz = nz;
    }

    blocks[ctr++] = rows;
    last_nz += irp[rows] - irp[rows-1];
    GET_MAX(max, last_nz)
    *max_nz = max;
    return ctr;
}

/**
 * Computes the dimension of the dynamic shared memory to be used in each block.
 *
 * With this implementation each block should use a memory of (local_nz*k) doubles. To give each one the same size,
 * 'max_nz' computed during 'get_csr_row_blocks()'.
 *
 * When max_nz*k is bigger than the maximum available shared memory in the device, halve the amount of memory.
 * The number of times this amount has been halved will be given to the blocks to know when to reduce.
 * */
 //TODO: manage when shmem is bigger than available
int get_shared_memory(int max, int k){
    int num = (max * 2)*sizeof(double);
    /*
    int halves = 0;

    while (num > MAX_SHARED_MEM) {
        num = (num+1)/2;
        halves++;
    }

    *round = halves+1;
    return num;*/

    if (num > MAX_SHARED_MEM) {
        return -1;
    } else {
        return num;
    }
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

__device__ Type warp_reduce(Type sum){
    // implementation of a logarithmic reduction with warp-level communication primitive
    for(int s = warpSize >> 1; s > 0; s >>= 1) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, s);
    }

    return sum;
}



