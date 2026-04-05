#ifndef NML_LINALG_H
#define NML_LINALG_H

#include "nml/types.h"

void zaxpy_variant(int n, const nml_complex *a, const nml_complex *x,
                   nml_complex *y);
void lower_tri_diag_isqrt_mult(int m, const double *D, nml_complex *L);
void diagonal_averaging(nml_complex *z, const nml_complex *S, int m);
void diagonal_averaging_full(nml_complex *z, const nml_complex *S, int m);
void tri_to_full(nml_complex *dest, const nml_complex *src, int ncols);
void pad_with_zeros(const nml_complex *A, nml_complex *A_padded, int nrows,
                    int ncols, int N);
void hermitian_conj(nml_complex *A, int nrows);

#endif
