#include "linalg.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/* y = x + a*conj(reverse(x)). */
void zaxpy_variant(const int n, const double complex *a, double complex *x, double complex *y){
    for(int i = 0; i<n; i++){
        y[i] = x[i] + (*a)*conj(x[n-1-i]);
    }
}

/* L <- L*sqrt(D^{-1}) where D is diagonal and L is lower triangular.
   This operation scales the columns of L.
 
    IN: 
        D: size m array with diagonal entries
        L: column major representation of L
 
    NOTE: D must be real
 */
void lower_tri_diag_isqrt_mult(const int m, const double *D, double complex *L){
    int row, col, index;
    index = 0;
    for(col = 0; col<m; col++){
        for(row = 0; row<m-col; row++){
            L[index + row] = L[index + row]/sqrt(D[col]);
        }
        index += (m - col);
    }
}

/*
    IN: 
        z: represents the first column of S
        S: column major representation of lower part of S
        m: dimension of S (S is m x m)
*/
void diagonal_averaging(double complex *z, double complex *S, const int m){
    int i, j, index;
    for(i = 0; i<m; i++){
        z[i] = 0;
        index = i;
        for(j = 0; j<m-i; j++){
            z[i] += S[index];
            index += (m - j);
        }
        z[i] = z[i]/(m-i);
    }
}

/* Computes the average along the diagonals in the lower triangular part of a 
   matrix S.
        z: represents the first column of S
        S: column major representation of S, full storage (so array of length m * m)
        m: dimension of S (S is m x m)
*/
void diagonal_averaging_full(double complex *z, double complex *S, const int m){
    int i, j, index;
    for(i = 0; i<m; i++){
        z[i] = 0;
        index = i;
        for(j = 0; j < m-i; j++){
            z[i] += S[index];
            index += (m + 1);
        }
        z[i] /= (m-i);
    }
}


void tri_to_full(double complex *dest, double complex *src, int ncols){
    int col, index_dest, index_src;
    index_dest = 0;
    index_src = 0;
    memset(dest, 0, ncols*ncols*sizeof(double complex));
    for(col = 0; col<ncols; col++){
       memcpy(dest + index_dest, src + index_src, (ncols - col)*sizeof(double complex));
       index_dest += ncols + 1;
       index_src += (ncols - col);
    }
}

/* Assumes column-major order. A is nrows x n_cols, A_padded is N x n_cols. */
void pad_with_zeros(double complex *A, double complex *A_padded, int nrows, int ncols, int N){
    int col, ii;
    memset(A_padded, 0, N*ncols*sizeof(double complex));          /* Set A_padded to zero */
    for(col = 0; col < ncols; col++){
        for(ii=0; ii<nrows; ii++)
            A_padded[col*N+ii] = A[col*nrows + ii]; 
    }
}

/* in-place hermitian conjugate of a matrix. Assumes column-major order */
void hermitian_conj(double complex *A, int nrows){
    int i, k;
    double complex temp;    
    /* k represents col, i represents row */
    for (k = 0; k < nrows; k++){
            A[k+k*nrows] = conj(A[k+k*nrows]);
            for (i = k + 1; i < nrows; i++){
                temp = A[i + k*nrows];                           
                A[i + k*nrows] = conj(A[k + i*nrows]);
                A[k + i*nrows] = conj(temp);
            }
        }
}
