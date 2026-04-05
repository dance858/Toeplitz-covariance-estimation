#include "diff.h"
#include "linalg.h"
#include "nml/platform/blas_lapack.h"
#include "utils.h"
#include <complex.h>
#include <fftw3.h>
#include <string.h>

void compute_derivatives_packed(NML_solver *solver)
{
    NML_workspace *w = &solver->work;
    int n = solver->n;
    int n_plus_one = solver->n_plus_one;
    int two_n_plus_one = solver->two_n_plus_one;
    int N = solver->N;

    /* DFT of chol_toep. */
    pad_with_zeros(w->full_chol_toep, w->R_DFT, n_plus_one, n_plus_one, N);
    fftw_execute_dft(w->plan_R_DFT, w->R_DFT, w->R_DFT);
    double complex one = 1;

    /* compute A = R*(R'*L), where R is lower triangular. Store it in w->RHL. */
    cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                n_plus_one, n_plus_one, &one, w->full_chol_toep, n_plus_one,
                w->RHL, n_plus_one);

    /* compute DFT of A */
    pad_with_zeros(w->RHL, w->A_DFT, n_plus_one, n_plus_one, N);
    fftw_execute_dft(w->plan_A_DFT, w->A_DFT, w->A_DFT);

    /* compute quantity that should be IFFT:ed to obtain the gradient */
    int i, k;
    double complex Rik, Aik;
    memset(w->grad_help, 0, N * sizeof(double complex));
    for (k = 0; k < n_plus_one; k++)
    {
        for (i = 0; i < N; i++)
        {
            Rik = w->R_DFT[i + k * N];
            Aik = w->A_DFT[i + k * N];
            w->grad_help[i] += (creal(Rik) * creal(Rik) + cimag(Rik) * cimag(Rik) -
                                creal(Aik) * creal(Aik) - cimag(Aik) * cimag(Aik));
        }
    }

    /* IFFT (not normalized) */
    fftw_execute(w->plan_grad_help);

    /* correct scaling */
    complex double alpha = 2.0 / N;
    cblas_zscal(N, &alpha, w->grad_help, 1);

    /* parse grad_help to obtain the true gradient */
    w->grad[0] = creal(w->grad_help[0]);
    for (i = 1; i < n_plus_one; i++)
    {
        w->grad[i] = creal(w->grad_help[i]);
        w->grad[i + n] = -cimag(w->grad_help[i]);
    }

    /* F = R_DFT*R_DFT^H, complex-valued Hermitian matrix stored as lower triangular */
    cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, N, n_plus_one, 1,
                w->R_DFT, N, 0, w->F, N);

    /* G = A_DFT*A_DFT^H, complex-valued Hermitian matrix stored as lower triangular */
    cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, N, n_plus_one, 1,
                w->A_DFT, N, 0, w->G, N);

    /* hess_help = F.*G^T + F^T.*G - F.*F^T. */
    for (k = 0; k < N; k++)
    {
        for (i = k; i < N; i++)
        {
            w->hess_help[i + k * N] =
                2 * (creal(w->F[i + k * N]) * creal(w->G[i + k * N]) +
                     cimag(w->F[i + k * N]) * cimag(w->G[i + k * N])) -
                creal(w->F[i + k * N]) * creal(w->F[i + k * N]) -
                cimag(w->F[i + k * N]) * cimag(w->F[i + k * N]);
            w->hess_help[k + i * N] = w->hess_help[i + k * N];
        }
    }

    /* hess_help = (F.*G^T + F^T.*G - F.*F^T)*W */
    fftw_execute_dft(w->plan_hess_help, w->hess_help, w->hess_help);
    hermitian_conj(w->hess_help, N);

    /* hess_help = W^H*(F.*G^T + F^T.*G - F.*F^T)*W */
    fftw_execute_dft(w->plan_hess_help, w->hess_help, w->hess_help);

    /* correct scaling: 4/N^2 */
    alpha = 4.0 / (N * N);
    cblas_zscal(N * N, &alpha, w->hess_help, 1);

    /* build the Hessian (k = column, i = row) */
    for (k = 0; k < n_plus_one; k++)
    { /* xx-block */
        for (i = k; i < n_plus_one; i++)
        {
            w->hess_packed[i + k * two_n_plus_one - k * (k + 1) / 2] =
                0.5 * (creal(w->hess_help[i + k * N]) +
                       creal(w->hess_help[(k + 1) * N - i]));
        }
    }
    w->hess_packed[0] = creal(w->hess_help[0]);

    for (k = 0; k < n_plus_one; k++)
    { /* yx-block */
        for (i = n_plus_one; i < two_n_plus_one; i++)
        {
            w->hess_packed[i + k * two_n_plus_one - k * (k + 1) / 2] =
                0.5 * (cimag(w->hess_help[(k + 1) * N - i + n]) -
                       cimag(w->hess_help[i + k * N - n]));
        }
    }

    for (k = n_plus_one; k < two_n_plus_one; k++)
    { /* yy-block */
        for (i = k; i < two_n_plus_one; i++)
        {
            w->hess_packed[i + k * two_n_plus_one - k * (k + 1) / 2] =
                0.5 * (creal(w->hess_help[i + (k - n) * N - n]) -
                       creal(w->hess_help[(k + 1 - n) * N - i + n]));
        }
    }
}
