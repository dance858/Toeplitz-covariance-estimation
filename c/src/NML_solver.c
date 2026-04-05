#include "NML_solver_internal.h"
#include "Newton.h"
#include "Timer.h"
#include "linalg.h"
#include "nml/levinson_durbin.h"
#include "nml/platform/blas_lapack.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Computes an initial guess as described in the paper.
    IN: Z_data: data matrix, size (n+1) x K stored in column-major order

    NOTE: After exit, w->chol_toep has been modified so
          T(x, y)^{-1} = w->chol_toep * w->chol_toep^H.
*/
static void init_guess(const double complex *Z_data, NML_solver *solver,
                       NML_result *result)
{
    NML_workspace *w = &solver->work;
    int n = solver->n;
    int n_plus_one = solver->n_plus_one;
    int i, status;

    /* Compute sample covariance. The lower triangular part is stored in a
       full matrix of size (n+1) x (n+1) */
    memset(w->sample_cov, 0, sizeof(double complex) * n_plus_one * n_plus_one);
    cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, n_plus_one, solver->K,
                1.0 / solver->K, Z_data, n_plus_one, 0.0, w->sample_cov, n_plus_one);

    /* Cholesky factorization of sample covariance. If the factorization fails
       because of numerical issues we add a small regularization. */
    status = 1;
    while (status != 0)
    {
        memcpy(w->L_full, w->sample_cov,
               sizeof(double complex) * n_plus_one * n_plus_one);
        status =
            LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'L', n_plus_one, w->L_full, n_plus_one);

        /* add small regularization to diagonal elements of sample cov */
        for (i = 0; i < n_plus_one; i++)
        {
            w->sample_cov[i + i * n_plus_one] += 1e-8;
        }
    }

    /* Diagonal averaging of sample covariance */
    diagonal_averaging_full(w->z, w->sample_cov, n + 1);

    /* Check if preliminary initial guess belongs to domain */
    status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, n);

    /* if status is equal to 1 it means that the initial guess does not belong
       to the domain. In this case we should compute an initial guess based on
       the circulant ML estimate. */
    if (status == 1)
    {
        result->diag_init_succeeded = 0;

        /* Copy Z_data since the in-place FFT below would modify the caller's data */
        int size = n_plus_one * solver->K;
        double complex *Z_copy = malloc(sizeof(double complex) * size);
        memcpy(Z_copy, Z_data, sizeof(double complex) * size);

        /* IFFT of data points, scaled correctly */
        fftw_plan plan_Z_data = fftw_plan_many_dft(
            1, &n_plus_one, solver->K, Z_copy, &n_plus_one, 1, n_plus_one, Z_copy,
            &n_plus_one, 1, n_plus_one, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute_dft(plan_Z_data, Z_copy, Z_copy);

        /* compute q of size (n+1) x 1, stored in z */
        memset(w->z, 0, sizeof(double complex) * n_plus_one);
        for (int k = 0; k < solver->K; k++)
        {
            for (i = 0; i < n_plus_one; i++)
            {
                w->z[i] += (creal(Z_copy[i + k * n_plus_one]) *
                                creal(Z_copy[i + k * n_plus_one]) +
                            cimag(Z_copy[i + k * n_plus_one]) *
                                cimag(Z_copy[i + k * n_plus_one]));
            }
        }

        /* compute FFT of q */
        fftw_plan plan_q =
            fftw_plan_dft_1d(n_plus_one, w->z, w->z, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan_q);

        /* scale with 1/(K*(n+1)^2). After this operation, w->z stores the first
           column of the circulant ML estimate. */
        complex double scale = 1.0 / (solver->K * n_plus_one * n_plus_one);
        cblas_zscal(n_plus_one, &scale, w->z, 1);

        status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, n);
        fftw_destroy_plan(plan_q);
        fftw_destroy_plan(plan_Z_data);
        free(Z_copy);
    }
    /* scale chol_toep so T(x, y)^{-1} = chol_toep * chol_toep^H */
    lower_tri_diag_isqrt_mult(n + 1, w->sigma2, w->chol_toep);

    w->xy[0] = creal(w->z[0]) / 2.0;
    for (i = 1; i < n_plus_one; i++)
    {
        w->xy[i] = creal(w->z[i]);
        w->xy[n + i] =
            -cimag(w->z[i]); /* minus sign follows from definition of T(x, y) */
    }
}

NML_solver *nml_new_solver(int n, double tol, double beta, double alpha, int max_iter)
{
    NML_solver *s = malloc(sizeof(*s));
    s->n = n;
    s->N = 2 * (n + 1);
    s->n_plus_one = n + 1;
    s->two_n_plus_one = 2 * n + 1;

    s->settings = (NML_settings){tol, beta, alpha, max_iter};

    NML_workspace *w = &s->work;
    int N = s->N;

    w->xy = malloc(sizeof(*w->xy) * (2 * n + 1));
    w->z = malloc(sizeof(*w->z) * (n + 1));

    w->grad_help = (double complex *) fftw_malloc(sizeof(double complex) * N);
    w->hess_help = (double complex *) fftw_malloc(sizeof(double complex) * N * N);
    w->grad = malloc(sizeof(*w->grad) * (2 * n + 1));
    w->F = malloc(sizeof(*w->F) * N * N);
    w->G = malloc(sizeof(*w->G) * N * N);
    w->hess_packed = malloc(sizeof(*w->hess_packed) * (n + 1) * (2 * n + 1));
    w->chol_hess_packed = malloc(sizeof(*w->chol_hess_packed) * (n + 1) * (2 * n + 1));
    w->hess_evals = malloc(sizeof(*w->hess_evals) * (2 * n + 1));

    w->full_chol_toep = malloc(sizeof(*w->full_chol_toep) * (n + 1) * (n + 1));
    w->chol_toep = malloc(sizeof(*w->chol_toep) * (n + 1) * (n + 2) / 2);
    w->sigma2 = malloc(sizeof(*w->sigma2) * (n + 1));
    w->RHL = malloc(sizeof(*w->RHL) * (n + 1) * (n + 1));
    w->L_full = malloc(sizeof(*w->L_full) * (n + 1) * (n + 1));
    w->sample_cov = malloc(sizeof(*w->sample_cov) * (n + 1) * (n + 1));

    w->neg_dir = malloc(sizeof(*w->neg_dir) * (2 * n + 1));

    w->R_DFT = (double complex *) fftw_malloc(sizeof(double complex) * N * (n + 1));
    w->A_DFT = (double complex *) fftw_malloc(sizeof(double complex) * N * (n + 1));

    w->plan_R_DFT =
        fftw_plan_many_dft(1, &N, n + 1, w->R_DFT, &N, 1, N, w->R_DFT, &N, 1, N,
                           FFTW_FORWARD, FFTW_ESTIMATE);
    w->plan_A_DFT =
        fftw_plan_many_dft(1, &N, n + 1, w->A_DFT, &N, 1, N, w->A_DFT, &N, 1, N,
                           FFTW_FORWARD, FFTW_ESTIMATE);
    w->plan_grad_help =
        fftw_plan_dft_1d(N, w->grad_help, w->grad_help, FFTW_BACKWARD, FFTW_ESTIMATE);
    w->plan_hess_help =
        fftw_plan_many_dft(1, &N, N, w->A_DFT, &N, 1, N, w->A_DFT, &N, 1, N,
                           FFTW_BACKWARD, FFTW_ESTIMATE);
    return s;
}

NML_result *nml_new_result(int n)
{
    NML_result *r = malloc(sizeof(*r));
    r->x = malloc(sizeof(double) * (n + 1));
    r->y = malloc(sizeof(double) * n);
    return r;
}

void nml_free_result(NML_result *result)
{
    if (!result) return;
    free(result->x);
    free(result->y);
    free(result);
}

void nml_free_solver(NML_solver *solver)
{
    if (!solver) return;
    NML_workspace *w = &solver->work;

    free(w->xy);
    free(w->z);

    fftw_free(w->grad_help);
    fftw_free(w->hess_help);
    free(w->grad);
    free(w->F);
    free(w->G);
    free(w->hess_packed);
    free(w->chol_hess_packed);
    free(w->hess_evals);

    free(w->full_chol_toep);
    free(w->chol_toep);
    free(w->sigma2);
    free(w->RHL);
    free(w->L_full);
    free(w->sample_cov);

    free(w->neg_dir);

    fftw_destroy_plan(w->plan_R_DFT);
    fftw_destroy_plan(w->plan_A_DFT);
    fftw_destroy_plan(w->plan_grad_help);
    fftw_destroy_plan(w->plan_hess_help);

    fftw_free(w->R_DFT);
    fftw_free(w->A_DFT);

    free(solver);
}

int nml_solve(NML_solver *solver, const double complex *Z, int K,
              NML_result *result, int verbose)
{
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start);

    solver->K = K;
    solver->verbose = verbose;

    result->diag_init_succeeded = 1;
    result->num_of_hess_chol_fails = 0;

    init_guess(Z, solver, result);
    Newton(solver, result);

    clock_gettime(CLOCK_MONOTONIC, &timer.end);
    result->solve_time = GET_ELAPSED_SECONDS(timer);

    return 0;
}
