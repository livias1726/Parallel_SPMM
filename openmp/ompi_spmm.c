#include <mpi.h>
#include "omp_utils.h"

/**
 * Hybrid MPI/OpenMP version of a matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 * */

//--------------------------------------------------------------------------------------------------------------------//
void broadcasting(MPI_Comm comm, int rank, int *n, int *k, double **x){
    MPI_Bcast(n, 1, MPI_INT, 0, comm);
    MPI_Bcast(k, 1, MPI_INT, 0, comm);
    alloc_struct(x, *n, *k);
    if (rank == 0) populate_multivector(*x, *n, *k);
    MPI_Bcast(*x, (*n) * (*k), MPI_DOUBLE, 0, comm);
}

void load_balancing_scatterv(int size, int *rows_load, int *o_irp, int m, int *ma, int *nza,
                             int **irp, int **m_disp, int **nz_disp){

    *irp = (int*)malloc((m+size)*sizeof(int)); // copy of IRP to simulate overlapping in scattering
    *m_disp = (int*)malloc(size*sizeof(int)); // displacements to scatter 'ma'
    *nz_disp = (int*)malloc(size*sizeof(int)); // displacements to scatter 'nza'
    malloc_handler(3, (void*[]){*irp, *m_disp, *nz_disp});

    int start_row, end_row, l_m, m_count = 0, j, count = 0;

    for(int i = 0; i < size; i++){
        start_row = rows_load[i];
        end_row = rows_load[i + 1];
        l_m = end_row - start_row;

        ma[i] += 1;

        if (i == 0) {
            (*nz_disp)[i] = 0;
            (*m_disp)[i] = 0;
        }else{
            (*nz_disp)[i] = (*nz_disp)[i-1] + nza[i-1];
            (*m_disp)[i] = (*m_disp)[i-1] + ma[i-1];
        }

        for (j = m_count; j <= m_count+l_m; j++) {
            (*irp)[count] = o_irp[j];
            count++;
        }
        m_count += l_m;
    }
}

void load_balancing_scatter(int size, int *rows_load, int *irp, int **ma, int **nza){

    *ma = (int*)malloc(size*sizeof(int)); // number of rows per process
    *nza = (int*)malloc(size*sizeof(int)); // number of nz per process
    malloc_handler(2, (void*[]){*ma, *nza});

    int start_row, end_row;

    for(int i = 0; i < size; i++){
        start_row = rows_load[i];
        end_row = rows_load[i + 1];

        (*ma)[i] = end_row - start_row; // local number of rows
        (*nza)[i] = irp[end_row] - irp[start_row]; // local number non-zeros
    }
}

void csr_load_balancing(MPI_Comm comm, int size, int rank, CSR* csr, CSR** l_csr, int **ma){
    // global params
    int *irp, *ja;
    double *as;
    // local params
    int l_m, l_nz, *l_irp, *l_ja;
    double* l_as;

    // Setup of arrays to scatter wrt the balancing of non-zeros among processes
    int *nza, *nz_disp, *m_disp, *rows_load;
    int m, *o_irp;

    if(rank == 0) {
        o_irp = csr->IRP; // original IRP: need to create a modified copy to simulate overlapping in MPI_Scatterv
        m = csr->M;
        ja = csr->JA;
        as = csr->AS;

        rows_load = nz_balancing(size, csr->NZ, o_irp, m);
        load_balancing_scatter(size, rows_load, o_irp, ma, &nza);
    }

    // Scattering
    MPI_Scatter(*ma, 1, MPI_INT, &l_m, 1, MPI_INT, 0, comm); // scatter local number of rows per process
    MPI_Scatter(nza, 1, MPI_INT, &l_nz, 1, MPI_INT, 0, comm); // scatter local number of nz per process

    l_irp = (int*)malloc((l_m+1) * sizeof(int));
    l_ja = (int*)malloc(l_nz * sizeof(int));
    l_as = (double*)malloc(l_nz * sizeof(double));
    malloc_handler(3, (void*[]){l_irp, l_ja, l_as});

    if (rank == 0) load_balancing_scatterv(size, rows_load, o_irp, m, *ma, nza, &irp, &m_disp, &nz_disp);

    MPI_Scatterv(irp, *ma, m_disp, MPI_INT, l_irp, l_m+1, MPI_INT, 0, comm); // scatter irp portion per process
    MPI_Scatterv(ja, nza, nz_disp, MPI_INT, l_ja, l_nz, MPI_INT, 0, comm); // scatter ja portion per process
    MPI_Scatterv(as, nza, nz_disp, MPI_DOUBLE, l_as, l_nz, MPI_DOUBLE, 0, comm); // scatter as portion per process

    *l_csr = (CSR*) malloc(sizeof(CSR));
    malloc_handler(1, (void*[]){*l_csr});
    (*l_csr)->M = l_m;
    (*l_csr)->NZ = l_nz;
    (*l_csr)->IRP = l_irp;
    (*l_csr)->JA = l_ja;
    (*l_csr)->AS = l_as;

    if (rank == 0) clean_up(5, (void*[]){rows_load, irp, nza, nz_disp, m_disp});
}

void ell_load_balancing(MPI_Comm comm, int size, int rank, ELL* ell, ELL** l_ell){
    //TODO
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
void product_csr(CSR* csr, int pid, int threads, double* x, int k, double* y){

    int rows = csr->M;
    int *irp = csr->IRP;
    int *ja = csr->JA;
    double *as = csr->AS;

    int j, z, first, last, start = irp[0];
    double *row_tmp, *col_tmp, a_j;

    #pragma omp parallel for num_threads(threads) private(j, z, row_tmp, col_tmp, a_j, first, last) \
                                                  shared(rows, irp, start, k, as, x, ja, y, pid) \
                                                  default(none)
    for (int i = 0; i < rows; i++) {
        first = irp[i] - start;
        last = irp[i+1] - start;

        row_tmp = &y[i * k]; //the respective Y's row to accumulate products on

        for (j = first; j < last; j++) { // iterate over the nz values in the row
            // load just once
            col_tmp = &x[ja[j]*k]; //the respective X's row index
            a_j = as[j]; //the respective NZ value

            // Loop unrolling
            if (k % 4 == 0) { // avoids to process values like '12' with a 3-level loop unroll
                for (z = 0; z < k; z += 4) {
                    row_tmp[z] += a_j * col_tmp[z];
                    row_tmp[z+1] += a_j * col_tmp[z+1];
                    row_tmp[z+2] += a_j * col_tmp[z+2];
                    row_tmp[z+3] += a_j * col_tmp[z+3];
                }
            } else if (k % 3 == 0) {
                for (z = 0; z < k; z += 3) {
                    row_tmp[z] += a_j * col_tmp[z];
                    row_tmp[z+1] += a_j * col_tmp[z+1];
                    row_tmp[z+2] += a_j * col_tmp[z+2];
                }
            } else {
                for (z = 0; z < k; z++) {
                    row_tmp[z] += a_j * col_tmp[z];
                }
            }
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

double* gather_result(MPI_Comm comm, int size, int rank, double *y_p, int m, int k, int l_m, int *ma){
    int *m_disp;
    double *y_complete;

    if(rank == 0) {
        m_disp = (int*)malloc(size*sizeof(int));
        y_complete = (double*) malloc(m * k * sizeof(double));
        malloc_handler(2, (void*[]){m_disp, y_complete});

        for (int i = 0; i < size; i++) {
            ma[i] = (ma[i]-1)*k;
            m_disp[i] = (i == 0) ? 0 : ma[i-1];
        }
    }

    MPI_Gatherv(y_p, l_m*k, MPI_DOUBLE, y_complete, ma, m_disp, MPI_DOUBLE, 0, comm);

    if(rank == 0) clean_up(2, (void*[]){ma, m_disp});

    return y_complete;
}

//-----------------------------------------------------------------------------------------Main
int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR *csr, *l_csr;
    ELL *ell;
    double flop, gflops_s, gflops_p, abs_err, rel_err, *x, *y_s, *y_p;
    int rank, size, m, l_m, n, nz, k, num_threads;
    struct timespec t1, t2;

    //------------------------------------------------- MPI setup ----------------------------------------------------//
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    //------------------------------------------ Setup by master process ---------------------------------------------//
    if (rank == 0) {
        // check the correct use of the program
        process_arguments(argc, argv, &f, &k, &num_threads);
        process_mm(&t, f);

        // read matrix from file
        Elem **elems = read_mm(f, &m, &n, &nz, t);
        fclose(f);

        // convert to wanted storage format
#ifdef ELLPACK
        //TODO: manage H-Ellpack
        ell = read_mm_ell(elems, m, n, nz);
        m = ell->M;
        nz = ell->NZ;
#ifdef DEBUG
        print_ell(ell);
#endif
#else
        csr = read_mm_csr(elems, m, n, nz);
        m = csr->M;
        n = csr->N;
        nz = csr->NZ;
#ifdef DEBUG
        print_csr(csr);
#endif
#endif

        flop = (double) 2 * k * nz;
    }

    broadcasting(comm, rank, &n, &k, &x);

    // ---------------------------------------------- Serial SpMM ---------------------------------------------- //
    if(rank == 0) {
        alloc_struct(&y_s, m, k);

        clock_gettime(CLOCK_MONOTONIC, &t1);
#ifdef ELLPACK
        serial_product_ell(ell, x, k, y_s);
#else
        serial_product_csr(csr, x, k, y_s);
#endif
        clock_gettime(CLOCK_MONOTONIC, &t2);
        gflops_s = GET_GFLOPS(t1, t2, flop);
    }

    //----------------------------------- Balancing and Communication ------------------------------------------- //
    int *ma;
#ifdef ELLPACK
    ell_load_balancing();
#else
    csr_load_balancing(comm, size, rank, csr, &l_csr, &ma);
    if (rank == 0) clean_up(4, (void*[]){csr->AS, csr->JA, csr->IRP, csr});
#endif

    //--------------------------------------- MPI+OpenMP SpMM ----------------------------------------------- //
    l_m = l_csr->M;
    alloc_struct(&y_p, l_m, k);

    MPI_Barrier(comm);
    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &t1);

#ifdef ELLPACK
    //TODO
#else
    product_csr(l_csr, rank, num_threads, x, k, y_p);
#endif

    MPI_Barrier(comm);
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t2);
        GET_GFLOPS(t1, t2, flop);
    }

#ifdef ELLPACK
    //TODO
#else
    clean_up(4, (void*[]){l_csr->AS, l_csr->JA, l_csr->IRP, l_csr});
#endif

    //--------------------------------------------------- Output
    double* y_complete = gather_result(comm, size, rank, y_p, m, k, l_m, ma);
    clean_up(2, (void*[]){x, y_p});

    if (rank == 0) {
#ifdef DEBUG
        // print results
        print_matrix(y_complete, m, k, "\nResult:\n");
#endif

#ifdef SAVE
        save_result(y_complete, csr->M, k);
#endif

        // check results
        get_errors(m, k, y_s, y_complete, &abs_err, &rel_err);
        clean_up(2, (void*[]){y_complete, y_s});

#ifdef PERFORMANCE
        fprintf(stdout, "%f", gflops_p);
#else
        fprintf(stdout, "Serial GFLOPS: %f\nParallel GFLOPS: %f\nAbsolute error: %.2e\nRelative error: %.2e\n",
                gflops_s, gflops_p, abs_err, rel_err);
#endif
    }

    MPI_Finalize();

    return 0;
}