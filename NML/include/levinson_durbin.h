#include <complex.h>

/* Cholesky factorization of the inverse of a positive definite Toeplitz matrix.
   
   IN:
        y: (y0, y1, ..., yp) represents the first column of the Toeplitz matrix
        L: allocated memory for (p+1)*(p+2)/2 doubles 
    sigma: allocated memory for p+1 doubles 

   OUT: 
       status: 1 if the factorization succeded, 0 otherwise.
*/
int lev_dur_real(double *y, double *L, double *sigma2, const int p);
int lev_dur_complex(double complex *y, double complex *L, double *sigma2, const int p);
