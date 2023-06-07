#include "../utils/headers/test.h"

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
    int i, j;
    float gflops_s, gflops_p;
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

    // create output csv
    sprintf(out_filepath, "perf/perf_omp_%s.csv", storage);
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

    while (fgets(name, NAME_MAX, mat_file)) {   // matrix
        name[strlen(name) - 1] = '\0';
        if (strcmp(name, "") == 0) continue;

        for (i = 0; i < NUM_K; i++) { // k
            // build command line
            sprintf(input, "%s %s %d %d", PROGRAM, name, ks[i], 0);

            printf("Execution: [%s]\n", input);

            // launch program
            pipe = popen(input, "r");
            if (pipe == NULL) {
                fprintf(stderr, "Cannot open pipe\n");
                exit(-1);
            }

            j = 1;
            while (fgets(output, PATH_MAX, pipe)) {
                tokenize_output_omp(output, &gflops_s, &gflops_p, &abs, &rel);
                // save on csv
                fprintf(out_file, "%s,%s,%d,%d,%f,%f,%.2e,%.2e\n", name, storage, ks[i], j, gflops_s, gflops_p, abs, rel);
                j++;
            }

            // close program
            if (pclose(pipe) == -1) {
                fprintf(stderr, "Cannot close pipe\n");
                exit(-1);
            }
        }
    }

    fclose(mat_file);
    fclose(out_file);

    return 0;
}
