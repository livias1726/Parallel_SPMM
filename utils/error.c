#include "utils.h"

/**
 * Computes the absolute and relative error of the parallel product
 * computation (wrt the serial one) using the infinity norm.
 * */
void get_errors(int rows, int cols, double* seq, double* par, double* abs, double* rel){
    int i, j, idx;
    double max_diff = 0.0, tmp_diff, tmp_norm, norm = 0.0;

    for (i = 0; i < rows; i++) {
        tmp_diff = 0.0;
        tmp_norm = 0.0;
        idx = i*cols;

        for (j = 0; j < cols; j++) {
            tmp_diff += fabs(par[idx + j] - seq[idx + j]);
            tmp_norm += fabs(seq[idx + j]);
        }

        if (tmp_diff > max_diff) max_diff = tmp_diff;
        if (tmp_norm > norm) norm = tmp_norm;
    }

    *abs = max_diff;
    *rel = max_diff/norm;
}

/*
double get_absolute_error(int dim, double* seq, double* par){

    int i;
    double err = 0.0;
    for(i=0; i<dim; i++){
        err += fabs(seq[i]-par[i]);
    }

    return err;
}

double get_relative_error(int dim, double abs, double* seq){

    int i;
    double norm = 0.0;
    for(i=0; i<dim; i++){
        norm += fabs(seq[i]);
    }

    return abs/norm;
}
 */