#ifndef NML_UTILS_H
#define NML_UTILS_H

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

void reverse_real(double *y, const int N);
void reverse_complex(double complex *y, const int N);
void print_my_vector_real(double *x, const int n);
void print_my_vector_complex(double complex *x, const int n);
void print_complex_lower_triangular_matrix(double complex *L, int nrows);
void print_real_lower_triangular_matrix(double *L, int nrows);
void print_complex_matrix(double complex *A, int nrows, int ncols);
void print_real_matrix(double *A, int nrows, int ncols);

#endif
