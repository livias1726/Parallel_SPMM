#include "utils.h"

double get_absolute_error(int dim, double* seq, double* par){

    int i;
    double err = 0;
    for(i=0; i<dim; i++){
        err += fabs(seq[i]-par[i]);
    }

    return err;
}

double get_relative_error(int dim, double abs, double* seq){

    int i;
    double norm = 0;
    for(i=0; i<dim; i++){
        norm += fabs(seq[i]);
    }

    return abs/norm;
}