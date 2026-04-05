#ifndef NML_LEVINSON_DURBIN_H
#define NML_LEVINSON_DURBIN_H

#include "nml/types.h"

/* Cholesky factorization of the inverse of a positive definite Toeplitz matrix.

   IN:
        y: (y0, y1, ..., yp) represents the first column of the Toeplitz matrix
        L: allocated memory for (p+1)*(p+2)/2 doubles
    sigma: allocated memory for p+1 doubles

   OUT:
       status: 1 if the factorization succeded, 0 otherwise.
*/
int lev_dur_real(const double *y, double *L, double *sigma2, int p);
int lev_dur_complex(const nml_complex *y, nml_complex *L, double *sigma2, int p);

#endif
