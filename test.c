#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_RUNS 5
#define NAME_MAX 20
#define NUM_STORAGE 2
#define NUM_K 7
#define PROGRAM "cmake-build-debug\\serial\\serial_mmp "
#define PATH "resources/files/"

#define get_gflops(t, k, nz) ((2*k*nz)/t)

void tokenize_output (char* output, double* start, double* end, int* nz) {
    char* tok = strtok(output, " ");
    *start = strtod(tok, NULL);

    tok = strtok(NULL, " ");
    *end = strtod(tok, NULL);

    tok = strtok(NULL, " ");
    *nz = strtol(tok, NULL, 10);
}

double elapsed_nanoseconds (double start, double end) {
    double secs, nanos = modf(start, &secs);
    double sece, nanoe = modf(end, &sece);

    double sec = sece - secs;
    double nano = nanoe - nanos;

    if (sec != 0) {
        nano += (sec* pow(10,9));
    }

    return nano;
}

int main(){

#ifndef PERFORMANCE
    fprintf(stderr, "Compilation needs PERFORMANCE flag");
    exit(-1);
#endif

    FILE *mat_file, *pipe, *out_file;
    int i, j, z, k, nz;
    double start, end, time, gflops = 0;
    char name[NAME_MAX], input[PATH_MAX], output[PATH_MAX], filepath[strlen(PATH) + NAME_MAX];
    char *path_idx, *name_idx, *storage_idx, *k_idx,
            *ks[NUM_K] = {" 3", " 4", " 8", " 12", " 16", " 32", " 64"},
            *storage[NUM_STORAGE] = {" csr", " ellpack"};

    // get list of matrix names
    if ((mat_file = fopen("matrices.txt", "r")) == NULL) {
        fprintf(stderr, "Cannot open matrices file\n");
        exit(-1);
    }
    strcpy(filepath, PATH); // initialize first half of filepath
    strcpy(input, PROGRAM); // add program name to input
    path_idx = filepath + strlen(PATH);

    //create csv for results
    if ((out_file = fopen("gflops.csv", "w+")) == NULL) {
        fprintf(stderr, "Cannot open output file\n");
        exit(-1);
    }
    fprintf(out_file, "Matrix, Storage Format, K, GFLOPS\n"); //header

    // run
    while (fgets(name, NAME_MAX, mat_file)) {
        // add matrix name to input
        name_idx = input + strlen(PROGRAM);
        name[strlen(name)-1] = '\0';
        strcpy(name_idx, name);
        // build path
        strcpy(path_idx, name);

        storage_idx = name_idx + strlen(name); // index of the copying point
        for (i=0; i<NUM_STORAGE; i++) {
            // add storage format to input
            strcpy(storage_idx, storage[i]);

            k_idx = storage_idx + strlen(storage[i]); // index of the copying point
            for (j=0; j<NUM_K; j++) {
                // add k value to input
                strcpy(k_idx, ks[j]);

                for (z=0; z<NUM_RUNS; z++) {
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

                    tokenize_output(output, &start, &end, &nz);
                    time = elapsed_nanoseconds(start, end);
                    k = strtol(ks[j], NULL, 10);
                    gflops += get_gflops(time, k, nz);
                }

                // retrieve average gflops
                gflops = gflops / NUM_RUNS;

                // save on csv
                fprintf(out_file, "%s, %s, %d, %f\n", name, storage[i], k, gflops);
            }

            gflops = 0;
        }
    }

    fclose(mat_file);
    fclose(out_file);

    return 0;
}
