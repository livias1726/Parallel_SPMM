#include "../utils/headers/test.h"

#define NUM_THREADS 40
#define PROGRAM "openmp/omp_spmm"
#define HEADER "Matrix,Storage Format,K,Num of Threads,Serial GFLOPS,Parallel GFLOPS,Absolute Err,Relative Err\n"

void tokenize_output_omp (char* output, float* gs, float* gp, Type* ae, Type* re) {
    char* tok = strtok(output, " ");
    char* tokens[4];
    int count = -1;

    while (tok != NULL && count < 4) {
        count++;
        tokens[count] = tok;
        tok = strtok(NULL, " ");
    }

    if (count == 3) {
        *gs = strtof(tokens[0], NULL);
        *gp = strtof(tokens[1], NULL);
        *ae = strtof(tokens[2], NULL);
        *re = strtof(tokens[3], NULL);
    }
}

int main(){

#ifndef PERFORMANCE
    fprintf(stderr, "Compilation needs PERFORMANCE flag");
    exit(-1);
#endif

    FILE *mat_file, *pipe, *out_file;
    int i, j, z;
    float tmp_gfs, tmp_gfp, gflops_s = 0, gflops_p = 0;
    Type abs, rel;
    char input[PATH_MAX];
    char out_filepath[PATH_MAX], output[PATH_MAX];
    char name[NAME_MAX];
    int ks[NUM_K] = {3, 4, 8, 12, 16, 32, 64};
    char* storage;

#ifdef ELLPACK
    storage = "ELL";
#else
    storage = "CSR";
#endif

    for (i = 0; i < NUM_K; i++) { // add k value to input
        for (j = 1; j <= NUM_THREADS; j++) { // add num_threads value to input
            //create csv for k and num_threads
            sprintf(out_filepath, "perf/perf_omp_%s_%d_%d.csv", storage, ks[i], j);
            if ((out_file = fopen(out_filepath, "w+")) == NULL) {
                fprintf(stderr, "Cannot open output file\n");
                exit(-1);
            }
            fprintf(out_file, HEADER);

            // get list of matrix names
            if ((mat_file = fopen("resources/matrices.txt", "r")) == NULL) {
                fprintf(stderr, "Cannot open matrices file (Error: %d)\n", errno);
                exit(-1);
            }

            while (fgets(name, NAME_MAX, mat_file)) {
                // add matrix name to input
                name[strlen(name) - 1] = '\0';
                if (strcmp(name, "") == 0) continue;

                // build command line
                sprintf(input, "%s %s %d %d", PROGRAM, name, ks[i], j);

                for (z = 0; z < NUM_RUNS; z++) {
                    printf("Execution %d: [%s]\n", z, input);

                    // launch program
                    pipe = popen(input, "r");
                    if (pipe == NULL) {
                        fprintf(stderr, "Cannot open pipe\n");
                        exit(-1);
                    }

                    while (fgets(output, PATH_MAX, pipe)) {}

                    // close program
                    if (pclose(pipe) == -1) {
                        fprintf(stderr, "Cannot close pipe\n");
                        exit(-1);
                    }

                    tokenize_output_omp(output, &tmp_gfs, &tmp_gfp, &abs, &rel);
                    gflops_s += tmp_gfs;
                    gflops_p += tmp_gfp;
                }

                // retrieve average gflops
                gflops_s /= NUM_RUNS;
                gflops_p /= NUM_RUNS;

                // save on csv
                fprintf(out_file, "%s,%s,%d,%d,%f,%f,%.2e,%.2e\n", name, storage, ks[i], j, gflops_s, gflops_p, abs, rel);

                gflops_s = 0;
                gflops_p = 0;
            }

            fclose(mat_file);
            fclose(out_file);
        }
    }

    return 0;
}
