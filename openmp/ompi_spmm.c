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

void prepare_inargs(int size, const int*nz_start, CSR* csr, int **l_m, int **l_nz, int **nz_disp, int** m_disp) {
    *l_m = (int*)malloc(size*sizeof(int));
    *m_disp = (int*)malloc(size*sizeof(int));
    *l_nz = (int*)malloc(size*sizeof(int));
    *nz_disp = (int*)malloc(size*sizeof(int));
    malloc_handler(4, (void*[]){*l_m, *m_disp, *l_nz, *nz_disp});

    int i, first, last;
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
        if (i == 0) {
            (*nz_disp)[i] = 0;
            (*m_disp)[i] = 0;
        }else{
            (*nz_disp)[i] = (*l_nz)[i-1];
            (*m_disp)[i] = (*l_m)[i-1];
        }
    }
}

void prepare_outargs(int size, int m, int k, const int *ma, double **y, int **mak, int **disp) {
    *y = (double*)malloc(m*k*sizeof(double));
    *mak = (int*)malloc(size*sizeof(int));
    *disp = (int*)malloc(size*sizeof(int));
    malloc_handler(3, (void*[]){*y, *mak, *disp});

    for(int i=0; i<size; i++){
        (*mak)[i] = ma[i]*k;
        (*disp)[i] = i == 0 ? 0 : (*mak)[i-1];
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
void product_csr(int rows, int last_nz, const int* irp, const int* ja, const double* as, const double* x, int k, double* y){
    int j, z, first, last, start = irp[0];
    double temp;

    #pragma omp parallel for private(j, z, temp, first, last) shared(k, y, rows, start, last_nz, irp, x, as, ja) default(none)
    for (int i = 0; i < rows; i++) {
        first = irp[i] - start;
        last = (i == rows-1) ? last_nz : irp[i+1];
        last -= start;

        for (z = 0; z < k; z++) {
            temp = 0.0;

            for (j = first; j < last; j++) {
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

void broadcast_communication(MPI_Comm comm, int rank, double **x, int *n, int *nz, int *k) {
    MPI_Bcast(n, 1, MPI_INT, 0, comm);
    MPI_Bcast(k, 1, MPI_INT, 0, comm);
    MPI_Bcast(nz, 1, MPI_INT, 0, comm);

    alloc_struct(x, *n, *k);
    if (rank == 0) populate_multivector(*x, *n, *k);
    MPI_Bcast(*x, (*n) * (*k), MPI_DOUBLE, 0, comm);
}

void scattering(int size, int rank, MPI_Comm comm, CSR* csr,
                int **ma, int *l_m, int *l_nz, int* last_nz, int **l_irp, int **l_ja, double **l_as){

    int *nz_start, *nza, *nz_disp, *m_disp;
    int *irp, *ja;
    double *as;

    if(rank == 0) {
        irp = csr->IRP;
        ja = csr->JA;
        as = csr->AS;

        nz_start = nz_balancing(size, csr->NZ, irp, csr->M); //compute load balancing
        prepare_inargs(size, nz_start, csr, ma, &nza, &nz_disp, &m_disp);
    }

    MPI_Scatter(*ma, 1, MPI_INT, l_m, 1, MPI_INT, 0, comm);
    MPI_Scatter(nza, 1, MPI_INT, l_nz, 1, MPI_INT, 0, comm);

    *l_irp = (int*)malloc((*l_m)*sizeof(int));
    *l_ja = (int*)malloc((*l_nz)*sizeof(int));
    *l_as = (double*)malloc((*l_nz)*sizeof(double));
    malloc_handler(3, (void*[]){*l_irp, *l_ja, *l_as});

    MPI_Scatterv(irp, *ma, m_disp, MPI_INT, *l_irp, *l_m, MPI_INT, 0, comm);
    MPI_Scatterv(ja, nza, nz_disp, MPI_INT, *l_ja, *l_nz, MPI_INT, 0, comm);
    MPI_Scatterv(as, nza, nz_disp, MPI_DOUBLE, *l_as, *l_nz, MPI_DOUBLE, 0, comm);

    int *lnz_a;
    if (rank == 0) {
        lnz_a = (int*) malloc(size*sizeof(int));
        malloc_handler(1, (void*[]){lnz_a});
        for (int i=0; i<size-1; i++) {
            lnz_a[i] = irp[(*ma)[i]];
        }
        lnz_a[size-1] = csr->NZ;
    }
    MPI_Scatter(lnz_a, 1, MPI_INT, last_nz, 1, MPI_INT, 0, comm);

    if (rank == 0) {
        clean_up(7, (void*[]){nz_start, nza, m_disp, nz_disp, as, ja, lnz_a});
    }
}

//-----------------------------------------------------------------------------------------Main
int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR *csr;
    ELL *ell;
    long time;
    double gflops_p, *x, *y;
    int rank, size, k, n, nz, num_threads;
    struct timespec t1, t2;

    //---------------------------- MPI setup
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    //----------------------------- OMP setup
    num_threads = (int)strtol(argv[3], NULL, 10);
    omp_set_num_threads(num_threads);

    //-------------------------------- Computation setup by master process
    // TODO: check if it's better to send each row from p0 to pX or allocate each local part directly in pX
    if (rank == 0) {
        // check the correct use of the program
        process_arguments(argc, argv, &f, &k);
        process_mm(&t, f);

        // convert to wanted storage format
#ifdef ELLPACK
        ell = read_mm_ell(f, t);
        m = ell->M;
        n = ell->N;
        nz = ell->NZ;
    #ifdef DEBUG
        print_ell(ell);
    #endif
#else
        csr = read_mm_csr(f, t);
        n = csr->N;
        nz = csr->NZ;
    #ifdef DEBUG
        print_csr(csr);
    #endif
#endif

        fclose(f);
    }

    //---------------------------------- Communication:
    // transmitting parameters separately is 3 times faster than creating and transmitting mini csr per process
    broadcast_communication(comm, rank, &x, &n, &nz, &k);

#ifdef ELLPACK
    //TODO
#else

    int l_m, l_nz, last_nz, *l_irp, *l_ja, *ma;
    double *l_as;
    scattering(size, rank, comm, csr, &ma, &l_m, &l_nz, &last_nz, &l_irp, &l_ja, &l_as);

    alloc_struct(&y, l_m, k);

    MPI_Barrier(comm);
    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &t1);

    product_csr(l_m, last_nz, l_irp, l_ja, l_as, x, k, y);

    MPI_Barrier(comm);
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_p = get_gflops(time, k, nz);
    }

    // CSR local cleanup
    clean_up(3, (void*[]){l_irp, l_ja, l_as});
#endif

    double* y_complete;
    int *mak, *mk_disp;

    if(rank == 0) {
        prepare_outargs(size, csr->M, k, ma, &y_complete, &mak, &mk_disp);
        free(ma);
    }

    MPI_Gatherv(y, l_m*k, MPI_DOUBLE, y_complete, mak, mk_disp, MPI_DOUBLE, 0, comm);
    clean_up(2, (void*[]){x, y});

    if (rank == 0) {
#ifdef DEBUG
        // print results
        print_matrix(y_complete, csr->M, k, "\nResult:\n");
#endif
#ifdef SAVE
        save_result(y_complete, csr->M, k);
#endif

        clean_up(4, (void*[]){csr, mak, mk_disp, y_complete});

#ifdef PERFORMANCE
        fprintf(stdout, "%f", gflops_p);
#else
        fprintf(stdout, "\nGFLOPS: %f\n", gflops_p);
#endif
    }

    MPI_Finalize();

    return 0;
}