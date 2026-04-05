#ifndef NML_LINALG_H
#define NML_LINALG_H

#include <complex.h>

void zaxpy_variant(int n, const double complex *a, const double complex *x,
                   double complex *y);
void lower_tri_diag_isqrt_mult(int m, const double *D, double complex *L);
void diagonal_averaging(double complex *z, const double complex *S, int m);
void diagonal_averaging_full(double complex *z, const double complex *S, int m);
void tri_to_full(double complex *dest, const double complex *src, int ncols);
void pad_with_zeros(const double complex *A, double complex *A_padded, int nrows,
                    int ncols, int N);
void hermitian_conj(double complex *A, int nrows);

#endif
