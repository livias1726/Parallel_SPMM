#include <mpi.h>
#include "omp_utils.h"

/**
 * ompi_spmm performs an hybrid MPI/OpenMP multithreaded version of a matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 *
 * This hybrid implementation is set to avoid cache conflicts between cpus
 * */

void process_arguments(int argc, char** argv, FILE **f, bool* ell_flag, int* k){
    if (argc < 5){
        fprintf(stderr, "Usage: %s [mm-filename] [storage-format] [k value] [num-threads]\n", argv[0]);
        exit(-1);
    }

    // create file path
    char path[PATH_MAX] = "../resources/files/";
    strcat(path, argv[1]);

    //check the correct opening of the matrix file
    *f = fopen(path, "r");
    if (*f == NULL) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        exit(-1);
    }

    // get k value and desired storage format
    *ell_flag = !strcmp(argv[2], "ellpack");
    *k = (int)strtol(argv[3], NULL, 10);
}

void prepare_process_arguments(int pid, int size, const int*nz_start, CSR* csr, CSR** l_csr){

    int l_nz, l_m, first = csr->IRP[nz_start[pid]], last;

    *l_csr = (CSR*) malloc(sizeof(CSR));

    if (pid == size-1) {
        l_m = csr->M-nz_start[pid];
        last = csr->NZ;
    } else {
        l_m = nz_start[pid + 1] - nz_start[pid]; // local number of rows
        last = csr->IRP[nz_start[pid + 1]];
    }

    l_nz = last - first; // local number non-zeros

    (*l_csr)->JA = (int*)malloc(l_nz*sizeof(int));
    (*l_csr)->AS = (double*)malloc(l_nz*sizeof(int));

    for(int i = first; i < last; i++){
        ((*l_csr)->JA)[i-first] = (csr->JA)[i];
        ((*l_csr)->AS)[i-first] = (csr->AS)[i];
    }

    (*l_csr)->M = l_m;
    (*l_csr)->N = csr->N;
    (*l_csr)->NZ = l_nz;
}

void get_mpi_csr(MPI_Datatype* MPI_CSR, int m, int nz){
    int lengths[STRUCT_DIM] = {1,1,1,m,nz,nz};
    MPI_Aint offsets[STRUCT_DIM];

    /*
    MPI_Datatype r_arr, j_arr, nz_arr;
    MPI_Type_contiguous(m, MPI_INT, &r_arr);
    MPI_Type_contiguous(nz, MPI_INT, &j_arr);
    MPI_Type_contiguous(nz, MPI_DOUBLE, &nz_arr);
    MPI_Type_commit(&r_arr);
    MPI_Type_commit(&j_arr);
    MPI_Type_commit(&nz_arr);
    */

    //MPI_Datatype types[STRUCT_DIM] = {MPI_INT, MPI_INT, MPI_INT, r_arr, j_arr, nz_arr};
    MPI_Datatype types[STRUCT_DIM] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};

    offsets[0] = offsetof(CSR, M);
    offsets[1] = offsetof(CSR, N);
    offsets[2] = offsetof(CSR, NZ);
    offsets[3] = offsetof(CSR, IRP);
    offsets[4] = offsetof(CSR, JA);
    offsets[5] = offsetof(CSR, AS);

    MPI_Type_create_struct(STRUCT_DIM, lengths, offsets, types, MPI_CSR);
}

void get_mpi_ell(MPI_Datatype* MPI_ELL, ELL* ell){
    int dim = (ell->M)*(ell->MAXNZ);

    int lengths[STRUCT_DIM] = {1,1,1,1,dim,dim};
    MPI_Aint offsets[STRUCT_DIM];

    /*
    MPI_Datatype j_arr, nz_arr;
    MPI_Type_contiguous(dim, MPI_INT, &j_arr);
    MPI_Type_contiguous(dim, MPI_DOUBLE, &nz_arr);
    MPI_Type_commit(&j_arr);
    MPI_Type_commit(&nz_arr);
     */
    /*
    MPI_Aint base;
    MPI_Get_address(&ell, &base);
    MPI_Get_address(&ell->M, &offsets[0]);
    MPI_Get_address(&ell->N, &offsets[1]);
    MPI_Get_address(&ell->NZ, &offsets[2]);
    MPI_Get_address(&ell->MAXNZ, &offsets[3]);
    MPI_Get_address(&ell->JA[0], &offsets[4]);
    MPI_Get_address(&ell->AS[0], &offsets[5]);

    offsets[0] = MPI_Aint_diff(offsets[0], base);
    offsets[1] = MPI_Aint_diff(offsets[1], base);
    offsets[2] = MPI_Aint_diff(offsets[2], base);
    offsets[3] = MPI_Aint_diff(offsets[3], base);
    offsets[4] = MPI_Aint_diff(offsets[4], base);
    offsets[5] = MPI_Aint_diff(offsets[5], base);
     */

    //MPI_Datatype types[STRUCT_DIM] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, j_arr, nz_arr};
    MPI_Datatype types[STRUCT_DIM] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};

    offsets[0] = offsetof(ELL, M);
    offsets[1] = offsetof(ELL, N);
    offsets[2] = offsetof(ELL, NZ);
    offsets[3] = offsetof(ELL, MAXNZ);
    offsets[4] = offsetof(ELL, JA);
    offsets[5] = offsetof(ELL, AS);

    MPI_Type_create_struct(STRUCT_DIM, lengths, offsets, types, MPI_ELL);
}

//---------------------------------------------------------------------------------------------------Product
/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param nz_start array of starting row indices for each thread
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * @param t1 pointer to first timeval structure
 * @param t2 pointer to second timeval structure
 * */
void product_csr(CSR mat, const double* x, int k, double* y){
    int j, z, rows = mat.M, cols = mat.N;
    double temp;

#pragma omp parallel for private(j, z, temp) shared(k, y, mat, rows, cols, x) default(none)
    for (int i = 0; i < rows; i++) {
        for (z = 0; z < k; z++) {
            temp = 0.0;

            for (j = 0; j < cols; j++) {
                temp += mat.AS[j]*(x[mat.JA[j]*k+z]);
            }

            y[i*k+z] = temp;
        }
    }
}

/**
 * Computes the product with A stored in a ELL format
 *
 * @param mat matrix in ellpack format
 * @param x multivector Nxk stored as 1D array
 * @param y receives product results Mxk stored as 1D array
 * @param t1 pointer to first timeval structure
 * @param t2 pointer to second timeval structure
 * */
void product_ell(ELL mat, const double* x, int k, double* y, struct timespec *t1, struct timespec *t2){
    int i, j, z, maxnz = mat.MAXNZ;
    double t, val;

    clock_gettime(CLOCK_MONOTONIC, t1);
    // TODO: version 1 -> to be optimized
#pragma omp parallel for schedule(guided) shared(k, maxnz, x, mat, y) private(z, t, j, val) default(none)
    for (i = 0; i < mat.M; i++) {
        for (z = 0; z < k; z++) { //TODO: check order of loops
            t = 0.0;

            for (j = 0; j < maxnz; j++) {
                val = mat.AS[i*maxnz+j];
                if (val == 0) { // if padding is reached break loop
                    break;
                }
                t += val*x[mat.JA[i*maxnz+j]*k+z];
            }

            y[i*k+z] = t;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, t2);
}

void gather_result(double* l_y, double** y, int m, int k, int l_m, int rank, MPI_Comm comm) {
    if (rank == 0) {
        *y = (double*)malloc(m*k*sizeof(double));
        MPI_Gather(l_y, l_m*k, MPI_DOUBLE, *y, l_m*k, MPI_DOUBLE, 0, comm);
    }  else {
        MPI_Gather(l_y, l_m*k, MPI_DOUBLE, *y, l_m*k, MPI_DOUBLE, 0, comm);
    }
}
//-----------------------------------------------------------------------------------------Main
int main(int argc, char** argv) {

    //----------------------------------------------- GLOBAL ----------------------------------------------------------
    MM_typecode t;
    FILE *f;
    CSR *csr, *ls_csr, *lr_csr;
    ELL *ell, *l_ell;
    long time;
    double gflops_s, gflops_p;
    double *x, *y;
    int rank, size, k, m, n, nz, num_threads;
    struct timespec t1, t2;
    bool ellpack;

    MPI_Status status;
    MPI_Init(&argc, &argv); //TODO

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    //------------------------ OpenMP
    // set number of threads
    num_threads = (int)strtol(argv[4], NULL, 10);
    omp_set_num_threads(num_threads);

    if (rank == 0) {
        // check the correct use of the program
        process_arguments(argc, argv, &f, &ellpack, &k);
        process_mm(&t, f);

        //------------------------------------------------ Pre-Processing
        // TODO: check if it's better to send each row from p0 to pX or allocate each local part directly in pX
        // convert to wanted storage format
        if (ellpack) {
            ell = read_mm_ell(f, t);
            m = ell->M;
            n = ell->N;
            nz = ell->NZ;
        #ifdef AUDIT
            print_ell(ell);
        #endif
        } else {
            csr = read_mm_csr(f, t);
            m = csr->M;
            n = csr->N;
            nz = csr->NZ;
        #ifdef AUDIT
            print_csr(csr);
        #endif
        }

        fclose(f);

        alloc_struct(&x, n, k);
        populate_multivector(x, n, k);

        MPI_Bcast(&m, 1, MPI_INT, 0, comm); //send num_rows from 0 to all
        MPI_Bcast(&n, 1, MPI_INT, 0, comm); //send num_cols from 0 to all
        MPI_Bcast(&nz, 1, MPI_INT, 0, comm); //send nz from 0 to all
        MPI_Bcast(x, n * k, MPI_DOUBLE, 0, comm); //send x from 0 to all
    }

    // Matrix struct creation
    MPI_Datatype MPI_CSR, MPI_ELL;
    if(ellpack){
        /* TODO
        get_mpi_ell(&MPI_ELL, ell);
        MPI_Type_commit(&MPI_ELL);
         */
    }else{
        get_mpi_csr(&MPI_CSR, m, nz);
        MPI_Type_commit(&MPI_CSR);
    }

    if(rank == 0){
        if (ellpack) {
            //TODO
        } else {
            int* nz_start = nz_balancing(size, csr->NZ, csr->IRP, csr->M); //compute load balancing
            // prepare local csr to send
            for (int i=1; i<size; i++) {
                printf("csr for process %d has ...", i);
                prepare_process_arguments(i, size, nz_start, csr, &ls_csr);
                printf(" address %p\n", ls_csr);
                MPI_Send(ls_csr, 1, MPI_CSR, i, 0, comm);
            }

            prepare_process_arguments(0, size, nz_start, csr, &lr_csr);
            free(nz_start);
        }
    } else {
        if (ellpack) {
            MPI_Recv(l_ell, 1, MPI_ELL, 0, 0, comm, &status);
        } else {
            // receive local csr
            lr_csr = (CSR*) malloc(sizeof(CSR));
            MPI_Recv(lr_csr, 1, MPI_CSR, 0, 0, comm, &status);
        }
    }

    MPI_Type_free(&MPI_CSR);
    alloc_struct(&y, lr_csr->M, k);

#ifdef AUDIT
    // print results
    print_matrix(x, n, k, "\nMultivector:\n");
#endif

    //------------------------------------------------- Computation
    // compute the product
    if (ellpack) {
        /*TODO:
        product_ell(*ell, x_p, k, y, &t1, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_p = get_gflops(time, k, nz);

        free(ell);
         */
    } else {
        if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &t1);
        product_csr(*lr_csr, x, k, y); //Product(l_mat, l_vec, l_res, l_rows, cols, l_cols, comm);
        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t2);
            time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
            gflops_p = get_gflops(time, k, csr->NZ);
        }
    }

    //save_result(y, m, k);
    double* y_complete;
    gather_result(y, &y_complete, csr->M, k, lr_csr->M, rank, comm);

    free(y);

    MPI_Finalize();

    //----------------------------------------------------- Single process --------------------------------------------
    free(x);
    free(csr);

#ifdef AUDIT
    // print results
    print_matrix(y, m, k, "\nResult:\n");
#endif

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f", gflops_s, gflops_p);
#else
    fprintf(stdout, "\nSerial GFLOPS: %f\nParallel GFLOPS: %f\n", gflops_s, gflops_p);
#endif

    return 0;
}