#include <mpi.h>
#include "omp_utils.h"

/**
 * ompi_spmm performs an hybrid MPI/OpenMP multithreaded version of a matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 *
 * This hybrid implementation is set to avoid cache conflicts between cpus
 * */
void process_arguments(int argc, char** argv, FILE **f, int* k){
    if (argc < 4){
        fprintf(stderr, "Usage: %s [mm-filename] [k value] [num-threads]\n", argv[0]);
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

    *k = (int)strtol(argv[2], NULL, 10);
}

void prepare_args(int size, const int*nz_start, CSR* csr, int **l_m, int **l_nz, int ***l_ja, double ***l_as) {
    *l_m = (int*)malloc(size*sizeof(int));
    *l_nz = (int*)malloc(size*sizeof(int));
    *l_ja = (int**)malloc(size*sizeof(int*));
    *l_as = (double**)malloc(size*sizeof(double*));

    int i, j, first, last;
    for(i=0; i<size; i++){
        first = csr->IRP[nz_start[i]];

        if (i == size-1) {
            (*l_m)[i] = (csr->M)-nz_start[i];
            last = csr->NZ;
        } else {
            (*l_m)[i] = nz_start[i + 1] - nz_start[i]; // local number of rows
            last = csr->IRP[nz_start[i + 1]];
        }

        (*l_nz)[i] = last - first; // local number non-zeros

        (*l_ja)[i] = (int*)malloc((*l_nz)[i]*sizeof(int));
        (*l_as)[i] = (double*)malloc((*l_nz)[i]*sizeof(double));

        for(j=first; j<last; j++){
            ((*l_ja)[i])[j-first] = (csr->JA)[j];
            ((*l_as)[i])[j-first] = (csr->AS)[j];
        }
    }
}

//---------------------------------------------------------------------------------------------------Product
/**
 * Computes the product with A stored in a CSR format
 *
 * @param mat matrix in csr format
 * @param x multivector Nxk stored as 1D array
 * @param k number of columns of x
 * @param y receives product results Mxk stored as 1D array
 * */
void product_csr(int rows, int cols, const int* ja, const double* as, const double* x, int k, double* y){
    int j, z;
    double temp;

#pragma omp parallel for private(j, z, temp) shared(k, y, rows, cols, x, as, ja) default(none)
    for (int i = 0; i < rows; i++) {
        for (z = 0; z < k; z++) {
            temp = 0.0;

            for (j = 0; j < cols; j++) {
                temp += as[j]*(x[ja[j]*k+z]);
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
    CSR *csr;
    ELL *ell;
    long time;
    double gflops_s, gflops_p;
    double *x, *y;
    int rank, size, k, m, n, nz, num_threads;
    int l_m, l_nz, *l_ja;
    double *l_as;
    struct timespec t1, t2;

    MPI_Status status;
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // set number of threads
    num_threads = (int)strtol(argv[3], NULL, 10);
    omp_set_num_threads(num_threads);

    if (rank == 0) {
        // check the correct use of the program
        process_arguments(argc, argv, &f, &k);
        process_mm(&t, f);

        //------------------------------------------------ Pre-Processing
        // TODO: check if it's better to send each row from p0 to pX or allocate each local part directly in pX
        // convert to wanted storage format
#ifdef ELLPACK
        ell = read_mm_ell(f, t);
        m = ell->M;
        n = ell->N;
        nz = ell->NZ;
    #ifdef AUDIT
        print_ell(ell);
    #endif
#else
        csr = read_mm_csr(f, t);
        m = csr->M;
        n = csr->N;
        nz = csr->NZ;
    #ifdef AUDIT
        print_csr(csr);
    #endif
#endif

        fclose(f);

        alloc_struct(&x, n, k);
        populate_multivector(x, n, k);
#ifdef AUDIT
        // print results
        print_matrix(x, n, k, "\nMultivector:\n");
#endif

        MPI_Bcast(&m, 1, MPI_INT, 0, comm); //send num_rows from 0 to all
        MPI_Bcast(&n, 1, MPI_INT, 0, comm); //send num_cols from 0 to all
        MPI_Bcast(&nz, 1, MPI_INT, 0, comm); //send nz from 0 to all
        MPI_Bcast(x, n * k, MPI_DOUBLE, 0, comm); //send x from 0 to all
    }

#ifdef ELLPACK
        //TODO
        product_ell(*ell, x_p, k, y, &t1, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_p = get_gflops(time, k, nz);

        free(ell);
#else
    if(rank == 0){
        int* nz_start = nz_balancing(size, csr->NZ, csr->IRP, csr->M); //compute load balancing

        int *ma, *nza, **jaa;
        double **asa;
        prepare_args(size, nz_start, csr, &ma, &nza, &jaa, &asa);

        // prepare local csr to send
        for (int i=1; i<size; i++) {
            MPI_Send(&(ma[i]), 1, MPI_INT, i, 0, comm);
            MPI_Send(&(nza[i]), 1, MPI_INT, i, 0, comm);
            MPI_Send(jaa[i], nza[i], MPI_INT, i, 0, comm);
            MPI_Send(asa[i], nza[i], MPI_DOUBLE, i, 0, comm);
        }
        // prepare local csr to use
        l_m = ma[0];
        l_ja = jaa[0];
        l_as = asa[0];
        free(nz_start);
    } else {
        // receive local csr
        MPI_Recv(&l_m, 1, MPI_INT, 0, 0, comm, &status);
        MPI_Recv(&l_nz, 1, MPI_INT, 0, 0, comm, &status);

        l_ja = (int*)malloc(l_nz*sizeof(int));
        l_as = (double*)malloc(l_nz*sizeof(double));

        MPI_Recv(l_ja, l_nz, MPI_INT, 0, 0, comm, &status);
        MPI_Recv(l_as, l_nz, MPI_DOUBLE, 0, 0, comm, &status);
    }

    alloc_struct(&y, l_m, k);

    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &t1);
    product_csr(l_m, n, l_ja, l_as, x, k, y);
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_p = get_gflops(time, k, csr->NZ);
    }
#endif

    //save_result(y, m, k);
    double* y_complete;
    gather_result(y, &y_complete, csr->M, k, l_m, rank, comm);

    free(y);
    MPI_Finalize();

    //----------------------------------------------------- Single process --------------------------------------------
    free(x);

#ifdef ELLPACK
    free(ell);
#else
    free(csr);
#endif

#ifdef AUDIT
    // print results
    print_matrix(y_complete, m, k, "\nResult:\n");
#endif

#ifdef PERFORMANCE
    fprintf(stdout, "%f %f", gflops_s, gflops_p);
#else
    fprintf(stdout, "\nSerial GFLOPS: %f\nParallel GFLOPS: %f\n", gflops_s, gflops_p);
#endif

    return 0;
}