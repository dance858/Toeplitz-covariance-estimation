#ifndef NML_LINALG_H
#define NML_LINALG_H

#include "nml/types.h"

/* y[i] = x[i] + a * conj(x[n-1-i]) */
void zaxpy_variant(int n, nml_complex a, const nml_complex *x, nml_complex *y);

/* L <- L * diag(1/sqrt(D)). L is lower triangular in packed column-major storage. */
void lower_tri_diag_isqrt_mult(int m, const double *D, nml_complex *L);

/* Average along diagonals of lower triangular part of S (packed storage). */
void diagonal_averaging(nml_complex *z, const nml_complex *S, int m);

/* Average along diagonals of lower triangular part of S (full storage). */
void diagonal_averaging_full(nml_complex *z, const nml_complex *S, int m);

/* Unpack lower triangular (packed column-major) into full matrix. */
void tri_to_full(nml_complex *dest, const nml_complex *src, int ncols);

/* Embed A (nrows x ncols) into A_padded (N x ncols) with zero padding. */
void pad_with_zeros(const nml_complex *A, nml_complex *A_padded, int nrows,
                    int ncols, int N);

/* In-place Hermitian conjugate (transpose + conjugate). Column-major order. */
void hermitian_conj(nml_complex *A, int nrows);

#endif
