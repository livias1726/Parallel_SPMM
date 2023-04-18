#include "headers/test.h"

void tokenize_output (char* output, float* gs, float* gp, Type* ae, Type* re, float* mb) {
    char* tok = strtok(output, " ");
    char* tokens[5];
    int count = -1;

    while (tok != NULL && count < 5) {
        count++;
        tokens[count] = tok;
        tok = strtok(NULL, " ");
    }

    if (count == 4) {
        *gs = strtod(tokens[0], NULL);
        *gp = strtod(tokens[1], NULL);
        *ae = strtod(tokens[2], NULL);
        *re = strtod(tokens[3], NULL);
        *mb = strtod(tokens[4], NULL);
    }
}
