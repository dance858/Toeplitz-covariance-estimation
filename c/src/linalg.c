#include "linalg.h"
#include <math.h>
#include <string.h>

void zaxpy_variant(int n, nml_complex a, const nml_complex *x, nml_complex *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i] + a * conj(x[n - 1 - i]);
    }
}

void lower_tri_diag_isqrt_mult(int m, const double *D, nml_complex *L)
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

void diagonal_averaging(nml_complex *z, const nml_complex *S, int m)
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

void diagonal_averaging_full(nml_complex *z, const nml_complex *S, int m)
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

void tri_to_full(nml_complex *dest, const nml_complex *src, int ncols)
{
    int index_dest = 0;
    int index_src = 0;
    memset(dest, 0, ncols * ncols * sizeof(nml_complex));
    for (int col = 0; col < ncols; col++)
    {
        memcpy(dest + index_dest, src + index_src,
               (ncols - col) * sizeof(nml_complex));
        index_dest += ncols + 1;
        index_src += (ncols - col);
    }
}

void pad_with_zeros(const nml_complex *A, nml_complex *A_padded, int nrows,
                    int ncols, int N)
{
    memset(A_padded, 0, N * ncols * sizeof(nml_complex));
    for (int col = 0; col < ncols; col++)
    {
        for (int i = 0; i < nrows; i++) A_padded[col * N + i] = A[col * nrows + i];
    }
}

void hermitian_conj(nml_complex *A, int nrows)
{
    for (int k = 0; k < nrows; k++)
    {
        A[k + k * nrows] = conj(A[k + k * nrows]);
        for (int i = k + 1; i < nrows; i++)
        {
            nml_complex temp = A[i + k * nrows];
            A[i + k * nrows] = conj(A[k + i * nrows]);
            A[k + i * nrows] = conj(temp);
        }
    }
}
