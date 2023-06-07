#include <unistd.h>
#include "../utils/headers/test.h"

#define PROGRAM "cuda/cu_spmm"
#define HEADER "Matrix,Storage Format,K,Serial GFLOPS,Parallel GFLOPS,Absolute Err,Relative Err\n"

void tokenize_output_cuda (char* output, float* gs, float* gp, Type* ae, Type* re) {
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
    int i;
    float gflops_s, gflops_p;
    Type abs, rel;
    char name[NAME_MAX], input[PATH_MAX], output[PATH_MAX], out_filepath[PATH_MAX];
    int ks[NUM_K] = {3, 4, 8, 12, 16, 32, 64};
    char* storage;

#ifdef ELLPACK
    storage = "ELL";
#else
    storage = "CSR";
#endif

    //create csv for k and num_threads
    sprintf(out_filepath, "perf/perf_cuda_%s.csv", storage);
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

    // run
    while (fgets(name, NAME_MAX, mat_file)) {
        // add matrix name to input
        name[strlen(name) - 1] = '\0';
        if (strcmp(name, "") == 0) continue;

        for (i = 0; i < NUM_K; i++) { // add k value to input
            // build command line
            sprintf(input, "%s %s %d", PROGRAM, name, ks[i]);

            printf("Execution: [%s]\n", input);

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

            tokenize_output_cuda(output, &gflops_s, &gflops_p, &abs, &rel);

            // save on csv
            fprintf(out_file, "%s,%s,%d,%f,%f,%.2e,%.2e\n", name, storage, ks[i], gflops_s, gflops_p, abs, rel);
        }
    }

    fclose(mat_file);
    fclose(out_file);

    return 0;
}
