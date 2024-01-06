#include <complex.h>

void zaxpy_variant(const int n, const double complex *a, double complex *x, double complex *y);
void lower_tri_diag_isqrt_mult(const int m, const double *D, double complex *L);
void diagonal_averaging(double complex *z, double complex *S, const int m);
void diagonal_averaging_full(double complex *z, double complex *S, const int m);
void tri_to_full(double complex *dest, double complex *src, int ncols);
void hermitian_conj(double complex *A, int nrows);