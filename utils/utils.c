#include "utils.h"

void read_csr(){

}

void write_csr(){

}

void read_ell(){

}

void write_ell(){

}

void get_mflops(time_t v, int dims){

    int num_ops = 2*dims*dims*dims;

    printf("MFLOPS: %f\n\n", num_ops/((double)v*pow(10,6)));
}