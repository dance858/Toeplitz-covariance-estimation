#ifndef NML_LAPACKE_COMPAT_H
#define NML_LAPACKE_COMPAT_H

#include <complex.h>

#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif

typedef int lapack_int;
typedef double _Complex lapack_complex_double;

lapack_int LAPACKE_zpotrf(int matrix_layout, char uplo, lapack_int n,
                          lapack_complex_double *a, lapack_int lda);

lapack_int LAPACKE_dpptrf(int matrix_layout, char uplo, lapack_int n, double *ap);

lapack_int LAPACKE_dspevx(int matrix_layout, char jobz, char range, char uplo,
                          lapack_int n, double *ap, double vl, double vu,
                          lapack_int il, lapack_int iu, double abstol, lapack_int *m,
                          double *w, double *z, lapack_int ldz, lapack_int *ifail);

#endif
