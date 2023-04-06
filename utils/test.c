#include "headers/test.h"

void tokenize_output (char* output, double* gs, double* gp, double* ae, double* re) {
    char* tok = strtok(output, " ");
    char* tokens[4];
    int count = -1;

    while (tok != NULL && count < 4) {
        count++;
        tokens[count] = tok;
        tok = strtok(NULL, " ");
    }

    if (count == 3) {
        *gs = strtod(tokens[0], NULL);
        *gp = strtod(tokens[1], NULL);
        *ae = strtod(tokens[2], NULL);
        *re = strtod(tokens[3], NULL);
    }
}
