#include "linalg.h"
#include <math.h>
#include <string.h>

/* y = x + a*conj(reverse(x)). */
void zaxpy_variant(int n, const double complex *a, const double complex *x,
                   double complex *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i] + (*a) * conj(x[n - 1 - i]);
    }
}

/* L <- L*sqrt(D^{-1}) where D is diagonal and L is lower triangular.
   This operation scales the columns of L.

    IN:
        D: size m array with diagonal entries
        L: column major representation of L

    NOTE: D must be real
 */
void lower_tri_diag_isqrt_mult(int m, const double *D, double complex *L)
{
    int index = 0;
    for (int col = 0; col < m; col++)
    {
        for (int row = 0; row < m - col; row++)
        {
            L[index + row] = L[index + row] / sqrt(D[col]);
        }
        index += (m - col);
    }
}

/*
    IN:
        z: output, represents the first column of S
        S: column major representation of lower part of S
        m: dimension of S (S is m x m)
*/
void diagonal_averaging(double complex *z, const double complex *S, int m)
{
    for (int i = 0; i < m; i++)
    {
        z[i] = 0;
        int index = i;
        for (int j = 0; j < m - i; j++)
        {
            z[i] += S[index];
            index += (m - j);
        }
        z[i] = z[i] / (m - i);
    }
}

/* Computes the average along the diagonals in the lower triangular part of a
   matrix S.
        z: output, represents the first column of S
        S: column major representation of S, full storage (array of length m * m)
        m: dimension of S (S is m x m)
*/
void diagonal_averaging_full(double complex *z, const double complex *S, int m)
{
    for (int i = 0; i < m; i++)
    {
        z[i] = 0;
        int index = i;
        for (int j = 0; j < m - i; j++)
        {
            z[i] += S[index];
            index += (m + 1);
        }
        z[i] /= (m - i);
    }
}

void tri_to_full(double complex *dest, const double complex *src, int ncols)
{
    int index_dest = 0;
    int index_src = 0;
    memset(dest, 0, ncols * ncols * sizeof(double complex));
    for (int col = 0; col < ncols; col++)
    {
        memcpy(dest + index_dest, src + index_src,
               (ncols - col) * sizeof(double complex));
        index_dest += ncols + 1;
        index_src += (ncols - col);
    }
}

/* Assumes column-major order. A is nrows x ncols, A_padded is N x ncols. */
void pad_with_zeros(const double complex *A, double complex *A_padded, int nrows,
                    int ncols, int N)
{
    memset(A_padded, 0, N * ncols * sizeof(double complex));
    for (int col = 0; col < ncols; col++)
    {
        for (int i = 0; i < nrows; i++) A_padded[col * N + i] = A[col * nrows + i];
    }
}

/* in-place hermitian conjugate of a matrix. Assumes column-major order */
void hermitian_conj(double complex *A, int nrows)
{
    for (int k = 0; k < nrows; k++)
    {
        A[k + k * nrows] = conj(A[k + k * nrows]);
        for (int i = k + 1; i < nrows; i++)
        {
            const double complex temp = A[i + k * nrows];
            A[i + k * nrows] = conj(A[k + i * nrows]);
            A[k + i * nrows] = conj(temp);
        }
    }
}
