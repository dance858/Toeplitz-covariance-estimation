#ifndef NML_UTILS_H
#define NML_UTILS_H

#include "nml/types.h"
#include <stdio.h>
#include <stdlib.h>

void reverse_real(double *y, int N);
void reverse_complex(nml_complex *y, int N);
void print_my_vector_real(double *x, int n);
void print_my_vector_complex(nml_complex *x, int n);
void print_complex_lower_triangular_matrix(nml_complex *L, int nrows);
void print_real_lower_triangular_matrix(double *L, int nrows);
void print_complex_matrix(nml_complex *A, int nrows, int ncols);
void print_real_matrix(double *A, int nrows, int ncols);

#endif
