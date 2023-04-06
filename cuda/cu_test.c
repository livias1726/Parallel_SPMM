#include <unistd.h> // sleep()
#include "../utils/headers/test.h"

#define PROGRAM "cuda/cu_spmm "

int main(){

#ifndef PERFORMANCE
    fprintf(stderr, "Compilation needs PERFORMANCE flag");
    exit(-1);
#endif

    FILE *mat_file, *pipe, *out_file;
    int i, j;
    double tmp_gfs, tmp_gfp, gflops_s = 0, gflops_p = 0, abs, rel;
    char name[NAME_MAX], input[PATH_MAX], output[PATH_MAX], filepath[strlen(PATH) + NAME_MAX], k_val[3];
    int ks[NUM_K] = {3, 4, 8, 12, 16, 32, 64};
    char* storage;

#ifdef ELLPACK
    storage = "ELL";
#else
    storage = "CSR";
#endif

    unsigned ptr1 = strlen(PROGRAM), ptr2;

    // get list of matrix names
    if ((mat_file = fopen("resources/matrices.txt", "r")) == NULL) {
        fprintf(stderr, "Cannot open matrices file (Error: %d)\n", errno);
        exit(-1);
    }
    strcpy(filepath, PATH); // initialize first half of filepath
    strcpy(input, PROGRAM); // add program name to input

    //create csv for results
    if ((out_file = fopen("perf_cuda.csv", "w+")) == NULL) {
        fprintf(stderr, "Cannot open output file\n");
        exit(-1);
    }
    fprintf(out_file, "Matrix,Storage Format,K,Serial GFLOPS,Parallel GFLOPS,Absolute Err,Relative Err\n"); //header

    // run
    while (fgets(name, NAME_MAX, mat_file)) {
        // add matrix name to input
        name[strlen(name)-1] = '\0';
        if (strcmp(name, "") == 0) continue;
        strcpy(&input[ptr1], name);

        // build path
        strcpy(&filepath[strlen(PATH)], name);
        ptr2 = ptr1 + strlen(name); // index of the copying point

        for (i=0; i<NUM_K; i++) { // add k value to input
            sprintf(k_val, " %d", ks[i]);
            strcpy(&input[ptr2], k_val);

            for (j=0; j<NUM_RUNS; j++) {
                printf("Execution %d: [%s]\n", j, input);

                // launch program
                pipe = popen(input, "r");
                if (pipe == NULL){
                    fprintf(stderr, "Cannot open pipe\n");
                    exit(-1);
                }

                while (fgets(output, PATH_MAX, pipe)){}

                // close program
                if (pclose(pipe) == -1) {
                    fprintf(stderr, "Cannot close pipe\n");
                    exit(-1);
                }

                tokenize_output(output, &tmp_gfs, &tmp_gfp, &abs, &rel);
                gflops_s += tmp_gfs;
                gflops_p += tmp_gfp;

                sleep(1); // gets better performances
            }

            // retrieve average gflops
            gflops_s = gflops_s / NUM_RUNS;
            gflops_p = gflops_p / NUM_RUNS;

            // save on csv
            fprintf(out_file, "%s,%s,%d,%f,%f,%.2e,%.2e\n", name, storage, ks[i], gflops_s, gflops_p, abs, rel);
        }

        gflops_s = 0;
        gflops_p = 0;
    }

    fclose(mat_file);
    fclose(out_file);

    return 0;
}
