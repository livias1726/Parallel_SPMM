#include <mpi.h>
#include "omp_utils.h"

/**
 * Hybrid MPI/OpenMP version of a matrix-multivector multiplication Y <- AX where
 * - A is a sparse matrix
 * - X is a multivector with given k columns
 * */

//--------------------------------------------------------------------------------------------------------------------//
//TODO: fix bug on l_nz
void csr_load_balancing(int size, CSR* csr, int *ma, int *nza, int *nz_disp, int *m_disp){
    int m = csr->M;
    int nz = csr->NZ;
    int *irp = csr->IRP;
    int *nz_start = nz_balancing(size, nz, irp, m);

    int start_row, end_row;
    for(int i = 0; i < size; i++){
        start_row = nz_start[i];
        end_row = nz_start[i + 1];

        ma[i] = end_row - start_row; // local number of rows
        nza[i] = irp[end_row] - irp[start_row]; // local number non-zeros

        if (i == 0) {
            nz_disp[i] = 0;
            m_disp[i] = 0;
        }else{
            nz_disp[i] = nza[i-1];
            m_disp[i] = ma[i-1];
        }

        printf("m_disp[%d] = %d\n", i, ma[i-1]);
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
void product_csr(CSR* csr, int threads, double* x, int k, double* y){

    int rows = csr->M;
    int *irp = csr->IRP;
    int *ja = csr->JA;
    double *as = csr->AS;

    int j, z, first, last, start = irp[0];
    double *row_tmp, *col_tmp, a_j;

    #pragma omp parallel for num_threads(threads) private(j, z, row_tmp, col_tmp, a_j, first, last) \
                                                  shared(rows, irp, start, k, as, x, ja, y) \
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

//-----------------------------------------------------------------------------------------Main
int main(int argc, char** argv) {

    MM_typecode t;
    FILE *f;
    CSR *csr;
    ELL *ell;
    double flop, gflops_s, gflops_p, abs_err, rel_err, *x, *y_s, *y_p;
    int rank, size;
    int k, num_threads;
    int m, n, nz, *irp, *ja;
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

        // read matrix from file
        Elem** elems = read_mm(f, &m, &n, &nz, t);
        fclose(f);

        // convert to wanted storage format
#ifdef ELLPACK
        //TODO: manage H-Ellpack
        ell = read_mm_ell(elems, m, n, nz);
        m = ell->M;
        n = ell->N;
        nz = ell->NZ;
    #ifdef DEBUG
        print_ell(ell);
    #endif
#else
        csr = read_mm_csr(elems, m, n, nz);
        m = csr->M;
        n = csr->N;
        nz = csr->NZ;
        irp = csr->IRP;
        ja = csr->JA;
        as = csr->AS;
    #ifdef DEBUG
        print_csr(csr);
    #endif
#endif

        flop = (double)2*k*nz;
        alloc_struct(&y_s, m, k);
    }

    //------------------------------------------- Communication ------------------------------------------- //
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    MPI_Bcast(&k, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nz, 1, MPI_INT, 0, comm);

    alloc_struct(&x, n, k);
    if (rank == 0) {
        populate_multivector(x, n, k);
        // ---------------------------------------------- Serial SpMM ---------------------------------------------- //
        clock_gettime(CLOCK_MONOTONIC, &t1);
#ifdef ELLPACK
        serial_product_ell(ell, x, k, y_s);
#else
        serial_product_csr(csr, x, k, y_s);
#endif
        clock_gettime(CLOCK_MONOTONIC, &t2);
        gflops_s = get_gflops(t1, t2, flop);
    }

    MPI_Bcast(x, n * k, MPI_DOUBLE, 0, comm);

#ifdef ELLPACK
    //TODO
#else
    int *ma, *nza, *nz_disp, *m_disp;
    if(rank == 0) {
        ma = (int*)malloc(size*sizeof(int));
        m_disp = (int*)malloc(size*sizeof(int));
        nza = (int*)malloc(size*sizeof(int));
        nz_disp = (int*)malloc(size*sizeof(int));
        malloc_handler(4, (void*[]){ma, m_disp, nza, nz_disp});

        csr_load_balancing(size, csr, ma, nza, nz_disp, m_disp);
    }

    int l_m, l_nz;

    MPI_Scatter(ma, 1, MPI_INT, &l_m, 1, MPI_INT, 0, comm); // scatter local number of rows per process
    MPI_Scatter(nza, 1, MPI_INT, &l_nz, 1, MPI_INT, 0, comm); // scatter local number of nz per process

    int* l_irp = (int*)malloc((l_m+1) * sizeof(int));
    int* l_ja = (int*)malloc(l_nz * sizeof(int));
    double* l_as = (double*)malloc(l_nz * sizeof(double));
    malloc_handler(3, (void*[]){l_irp, l_ja, l_as});

    MPI_Scatterv(irp, ma, m_disp, MPI_INT, l_irp, l_m+1, MPI_INT, 0, comm); // scatter irp portion per process
    MPI_Scatterv(ja, nza, nz_disp, MPI_INT, l_ja, l_nz, MPI_INT, 0, comm); // scatter ja portion per process
    MPI_Scatterv(as, nza, nz_disp, MPI_DOUBLE, l_as, l_nz, MPI_DOUBLE, 0, comm); // scatter as portion per process

    if (rank == 0) clean_up(2, (void*[]){nza, nz_disp});

    //--------------------------------------- MPI+OpenMP SpMM ----------------------------------------------- //
    alloc_struct(&y_p, l_m, k);
    CSR* l_csr = (CSR*) malloc(sizeof(CSR));
    malloc_handler(1, (void*[]){l_csr});

    l_csr->M = l_m;
    l_csr->NZ = l_nz;
    l_csr->IRP = l_irp;
    l_csr->JA = l_ja;
    l_csr->AS = l_as;

    MPI_Barrier(comm);
    if (rank == 0) clock_gettime(CLOCK_MONOTONIC, &t1);

    product_csr(l_csr, num_threads, x, k, y_p);

    MPI_Barrier(comm);
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t2);
        gflops_p = get_gflops(t1, t2, flop);
    }

    // CSR local cleanup
    clean_up(4, (void*[]){l_irp, l_ja, l_as, l_csr});
#endif

    //--------------------------------------------------- Output
    double* y_complete;

    if(rank == 0) {
        y_complete = (double*) malloc(m * k * sizeof(double));
        malloc_handler(1, (void*[]){y_complete});

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
        abs_err = get_absolute_error(m*k, y_s, y_complete);
        rel_err = get_relative_error(m*k, abs_err, y_s);

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