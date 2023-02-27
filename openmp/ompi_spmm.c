#include <mpi.h>
#include "omp_utils.h"

/**
 * ompi_spmm performs an hybrid MPI/OpenMP version of a matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 * */

void csr_load_balancing(int size, CSR* csr,
                        int **ma, int **nza, int **nz_disp, int **m_disp, int **lnz_a){

    int m = csr->M;
    int nz = csr->NZ;
    int* irp = csr->IRP;

    int *nz_start = nz_balancing(size, csr->NZ, irp, csr->M);

    *lnz_a = (int*) malloc(size*sizeof(int));
    *ma = (int*)malloc(size*sizeof(int));
    *m_disp = (int*)malloc(size*sizeof(int));
    *nza = (int*)malloc(size*sizeof(int));
    *nz_disp = (int*)malloc(size*sizeof(int));
    malloc_handler(5, (void*[]){*lnz_a, *ma, *m_disp, *nza, *nz_disp});

    int i, first_nz, last_nz, start_row, end_row;
    for(i=0; i<size; i++){
        start_row = nz_start[i];
        end_row = nz_start[i + 1];

        first_nz = irp[start_row];

        if (i == size-1) {
            (*ma)[i] = m-start_row;
            last_nz = nz;
            (*lnz_a)[i] = nz;
        } else {
            (*ma)[i] = end_row - start_row; // local number of rows
            last_nz = irp[end_row];
            (*lnz_a)[i] = irp[(*ma)[i]];
        }

        (*nza)[i] = last_nz - first_nz; // local number non-zeros

        if (i == 0) {
            (*nz_disp)[i] = 0;
            (*m_disp)[i] = 0;
        }else{
            (*nz_disp)[i] = (*nza)[i-1];
            (*m_disp)[i] = (*ma)[i-1];
        }
    }

    free(nz_start);
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
void product_csr(CSR* csr, int last_nz, const double* x, int k, double* y){
    int j, z, first, last, start = csr->IRP[0], rows = csr->M;
    double temp;

    #pragma omp parallel for private(j, z, temp, first, last) shared(csr, k, y, rows, start, last_nz, x) default(none)
    for (int i = 0; i < rows; i++) {
        first = csr->IRP[i] - start;
        last = (i == rows-1) ? last_nz : csr->IRP[i+1];
        last -= start;

        for (z = 0; z < k; z++) {
            temp = 0.0;

            for (j = first; j < last; j++) {
                temp += csr->AS[j]*(x[csr->JA[j]*k+z]);
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
        for (z = 0; z < k; z++) {
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

//-----------------------------------------------------------------------------------------Main
int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;

    CSR *csr;
    ELL *ell;

    long time;
    double gflops_s, gflops_p, abs_err, rel_err;
    double *x, *y_s, *y_p;
    int rank, size;
    int k, num_threads;
    int n, nz, *irp, *ja;
    double *as;
    struct timespec t1, t2;

    //---------------------------- MPI setup
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    //-------------------------------- Computation setup by master process
    if (rank == 0) {
        // check the correct use of the program
        process_arguments(argc, argv, &f, &k, &num_threads);
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
        irp = csr->IRP;
        ja = csr->JA;
        as = csr->AS;
    #ifdef DEBUG
        print_csr(csr);
    #endif
#endif

        fclose(f);
    }

    //---------------------------------- Broadcasting
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    MPI_Bcast(&k, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nz, 1, MPI_INT, 0, comm);

    alloc_struct(&x, n, k);
    if (rank == 0) populate_multivector(x, n, k);
    MPI_Bcast(x, n * k, MPI_DOUBLE, 0, comm);

#ifdef ELLPACK
    //TODO
#else

    int *ma, *nza, *lnz_a, *nz_disp, *m_disp;
    if(rank == 0) csr_load_balancing(size, csr, &ma, &nza, &nz_disp, &m_disp, &lnz_a);

    int l_m, l_nz, last_nz, *l_irp, *l_ja;
    double *l_as;

    //--------------------------------------------- Scattering
    MPI_Scatter(ma, 1, MPI_INT, &l_m, 1, MPI_INT, 0, comm); // scatter local number of rows per process
    MPI_Scatter(nza, 1, MPI_INT, &l_nz, 1, MPI_INT, 0, comm); // scatter local number of nz per process
    MPI_Scatter(lnz_a, 1, MPI_INT, &last_nz, 1, MPI_INT, 0, comm); // scatter last nz to compute per process

    l_irp = (int*)malloc(l_m * sizeof(int));
    l_ja = (int*)malloc(l_nz * sizeof(int));
    l_as = (double*)malloc(l_nz * sizeof(double));
    malloc_handler(3, (void*[]){l_irp, l_ja, l_as});

    MPI_Scatterv(irp, ma, m_disp, MPI_INT, l_irp, l_m, MPI_INT, 0, comm); // scatter irp portion per process
    MPI_Scatterv(ja, nza, nz_disp, MPI_INT, l_ja, l_nz, MPI_INT, 0, comm); // scatter ja portion per process
    MPI_Scatterv(as, nza, nz_disp, MPI_DOUBLE, l_as, l_nz, MPI_DOUBLE, 0, comm); // scatter as portion per process

    if (rank == 0) clean_up(3, (void*[]){nza, nz_disp, lnz_a});

    //--------------------------------------- Prepare multiplication
    alloc_struct(&y_p, l_m, k);
    CSR* l_csr = (CSR*) malloc(sizeof(CSR));
    l_csr->M = l_m;
    l_csr->NZ = l_nz;
    l_csr->IRP = l_irp;
    l_csr->JA = l_ja;
    l_csr->AS = l_as;

    MPI_Barrier(comm);
    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &t1);

    product_csr(l_csr, last_nz, x, k, y_p);

    MPI_Barrier(comm);
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t2);
        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_p = get_gflops(time, k, nz);

        alloc_struct(&y_s, csr->M, k);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        serial_product_csr(csr, x, k, y_s);
        clock_gettime(CLOCK_MONOTONIC, &t2);

        time = get_elapsed_nano(t1.tv_sec, t1.tv_nsec, t2.tv_sec, t2.tv_nsec);
        gflops_s = get_gflops(time, k, nz);
    }

    // CSR local cleanup
    clean_up(4, (void*[]){l_irp, l_ja, l_as, l_csr});
#endif

    //--------------------------------------------------- Output
    double* y_complete;

    if(rank == 0) {
        y_complete = (double*) malloc(csr->M * k * sizeof(double));
        for (int i = 0; i < size; i++) {
            ma[i] = ma[i]*k;
            m_disp[i] = (i == 0) ? 0 : ma[i-1];
        }
    }

    MPI_Gatherv(y_p, l_m*k, MPI_DOUBLE, y_complete, ma, m_disp, MPI_DOUBLE, 0, comm);
    clean_up(2, (void*[]){x, y_p});

    if (rank == 0) {
#ifdef DEBUG
        // print results
        print_matrix(y_complete, csr->M, k, "\nResult:\n");
#endif
#ifdef SAVE
        save_result(y_complete, csr->M, k);
#endif
        abs_err = get_absolute_error(csr->M*k, y_s, y_complete);
        rel_err = get_relative_error(csr->M*k, abs_err, y_s);

        clean_up(8, (void*[]){irp, as, ja, csr, ma, m_disp, y_complete, y_s});

#ifdef PERFORMANCE
        fprintf(stdout, "%f", gflops_p);
#else
        fprintf(stdout, "Serial GFLOPS: %f\nParallel GFLOPS: %f\nAbsolute error: %f\nRelative error: %f\n",
                gflops_s, gflops_p, abs_err, rel_err);
#endif
    }

    MPI_Finalize();

    return 0;
}