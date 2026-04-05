#include "Newton.h"
#include "diff.h"
#include "linalg.h"
#include "nml/levinson_durbin.h"
#include "nml/platform/blas_lapack.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

/* Evaluates the objective function
   f(x, y) = log det T(x, y) + Tr( T(x, y)^{-1} S).

   NOTE: Before this function is called, chol_toep and sigma2 must have been
   computed:
            T(x, y)^{-1} = chol_toep*diag(sigma2)^{-1}*chol_toep^H.
*/
static double compute_obj(NML_workspace *w, int n_plus_one)
{
    double obj = 0;
    nml_complex one = 1;

    /* compute w->RHL = R^H*L, where R is lower triangular and L is treated as a
       full matrix (despite L being lower triangular). */
    memcpy(w->RHL, w->L_full, sizeof(nml_complex) * n_plus_one * n_plus_one);
    tri_to_full(w->full_chol_toep, w->chol_toep, n_plus_one);
    cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit,
                n_plus_one, n_plus_one, &one, w->full_chol_toep, n_plus_one, w->RHL,
                n_plus_one);

    /* compute Tr(T(x, y)^{-1} S) */
    obj += cblas_dznrm2(n_plus_one * n_plus_one, w->RHL, 1);
    obj *= obj;

    /* compute log det T(x, y) */
    for (int i = 0; i < n_plus_one; i++)
    {
        obj += log(w->sigma2[i]);
    }
    return obj;
}

/* Convergence if Newton decrement < tol */
static int has_converged(NML_workspace *w, int two_n_plus_one, double tol)
{
    w->newton_dec = cblas_ddot(two_n_plus_one, w->grad, 1, w->neg_dir, 1);
    w->grad_norm = cblas_dnrm2(two_n_plus_one, w->grad, 1);
    return (w->newton_dec < tol);
}

/* Computes the step size.

    NOTE: When this function is called, w->neg_dir must contain the NEGATIVE
          search direction.
*/
static void compute_stepsize(NML_workspace *w, int n, int n_plus_one,
                             int two_n_plus_one, double beta, double alpha)
{
    w->step_size = 1 / beta; /* corresponds to initial step size 1 */
    int status, i;

    /* backtrack to ensure that the new iterate is in dom f */
    do
    {
        w->step_size *= beta;

        w->z[0] = 2 * (w->xy[0] - w->step_size * w->neg_dir[0]);
        for (i = 1; i < n_plus_one; i++)
        {
            w->z[i] = (w->xy[i] - w->step_size * w->neg_dir[i]) -
                      (w->xy[n + i] - w->step_size * w->neg_dir[n + i]) * NML_I;
        }

        /* status equal to 1 indicates that the factorization failed */
        status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, n);

    } while (status == 1);

    /* scale w->chol_toep so T(x, y)^{-1} = (w->chol_toep)*(w->chol_toep)^H */
    lower_tri_diag_isqrt_mult(n_plus_one, w->sigma2, w->chol_toep);

    /* backtrack until descent condition is satisfied */
    double dir_der = -cblas_ddot(two_n_plus_one, w->grad, 1, w->neg_dir, 1);

    while (1)
    {
        w->new_obj = compute_obj(w, n_plus_one);

        if (w->new_obj < w->obj + alpha * w->step_size * dir_der)
        {
            break;
        }

        w->step_size *= beta;
        w->z[0] = 2 * (w->xy[0] - w->step_size * w->neg_dir[0]);
        for (i = 1; i < n_plus_one; i++)
        {
            w->z[i] = (w->xy[i] - w->step_size * w->neg_dir[i]) -
                      (w->xy[n + i] - w->step_size * w->neg_dir[n + i]) * NML_I;
        }
        status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, n);
        lower_tri_diag_isqrt_mult(n_plus_one, w->sigma2, w->chol_toep);
    }
    w->obj = w->new_obj;
}

void Newton(NML_solver *solver, NML_result *result)
{
    NML_workspace *w = &solver->work;
    int n = solver->n;
    int n_plus_one = solver->n_plus_one;
    int two_n_plus_one = solver->two_n_plus_one;
    double tol = solver->settings.tol;
    double beta = solver->settings.beta;
    double alpha = solver->settings.alpha;
    int max_iter = solver->settings.max_iter;

    int i, k, info;
    int num_evals_found;
    int unused_ifail;
    double unused_eigvec;

    /* Compute objective value */
    w->obj = compute_obj(w, n_plus_one);

    if (solver->verbose)
    {
        printf(" iter | objective      | grad norm    | newton dec   | step size\n");
        printf("------+----------------+--------------+--------------+----------\n");
    }

    for (i = 0; i < max_iter; i++)
    {
        /* compute gradient and Hessian */
        compute_derivatives_packed(solver);

        /* Cholesky factorization of Hessian, modify it if necessary */
        while (1)
        {
            memcpy(w->chol_hess_packed, w->hess_packed,
                   sizeof(double) * n_plus_one * two_n_plus_one);
            info = LAPACKE_dpptrf(LAPACK_COL_MAJOR, 'L', two_n_plus_one,
                                  w->chol_hess_packed);
            if (info == 0)
            {
                break;
            }

            result->num_of_hess_chol_fails += 1;

            /* compute eigenvalues of packed matrix (overwrites, so copy first) */
            memcpy(w->chol_hess_packed, w->hess_packed,
                   sizeof(double) * n_plus_one * two_n_plus_one);
            info = LAPACKE_dspevx(LAPACK_COL_MAJOR, 'N', 'A', 'L', two_n_plus_one,
                                  w->chol_hess_packed, 0.0, 0.0, 0.0, 0.0, -1.0,
                                  &num_evals_found, w->hess_evals, &unused_eigvec,
                                  1.0, &unused_ifail);

            w->hess_evals[0] = MIN(-0.1, w->hess_evals[0]) * 1.05;
            /* add multiple of the identity to Hessian */
            for (k = 0; k < two_n_plus_one; k++)
            {
                w->hess_packed[k * (two_n_plus_one + 1) - k * (k + 1) / 2] -=
                    w->hess_evals[0];
            }
        }

        /* compute Newton direction. After this step the NEGATIVE of the
           direction is stored in w->neg_dir. */
        memcpy(w->neg_dir, w->grad, sizeof(double) * two_n_plus_one);
        cblas_dtpsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    two_n_plus_one, w->chol_hess_packed, w->neg_dir, 1);
        cblas_dtpsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                    two_n_plus_one, w->chol_hess_packed, w->neg_dir, 1);

        if (has_converged(w, two_n_plus_one, tol))
        {
            if (solver->verbose)
            {
                printf(" %4d | %14.6e | %12.4e | %12.4e |         -\n", i, w->obj,
                       w->grad_norm, w->newton_dec);
            }
            break;
        }

        /* compute step size. chol_toep and w->obj are updated. */
        compute_stepsize(w, n, n_plus_one, two_n_plus_one, beta, alpha);

        if (solver->verbose)
        {
            printf(" %4d | %14.6e | %12.4e | %12.4e | %9.4f\n", i, w->obj,
                   w->grad_norm, w->newton_dec, w->step_size);
        }

        /* update iterate */
        cblas_daxpy(two_n_plus_one, -w->step_size, w->neg_dir, 1, w->xy, 1);
    }

    /* store result */
    result->obj = w->obj;
    result->grad_norm = w->grad_norm;
    result->iter = i;
    memcpy(result->x, w->xy, sizeof(double) * (n + 1));
    memcpy(result->y, w->xy + n + 1, sizeof(double) * n);
}
