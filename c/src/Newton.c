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
static double compute_obj(NML_work *w)
{
    double obj = 0;
    const double complex one = 1;

    /* compute w->RHL = R^H*L, where R is lower triangular and L is treated as a
       full matrix (despite L being lower triangular). */
    memcpy(w->RHL, w->L_full,
           sizeof(double complex) * (w->n_plus_one) * (w->n_plus_one));
    tri_to_full(w->full_chol_toep, w->chol_toep, w->n_plus_one);
    cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasNonUnit,
                w->n_plus_one, w->n_plus_one, &one, w->full_chol_toep, w->n_plus_one,
                w->RHL, w->n_plus_one);

    /* compute Tr(T(x, y)^{-1} S) */
    obj += cblas_dznrm2((w->n_plus_one) * (w->n_plus_one), w->RHL, 1);
    obj *= obj;

    /* compute log det T(x, y) */
    for (int i = 0; i < w->n_plus_one; i++)
    {
        obj += log((w->sigma2)[i]);
    }
    return obj;
}

/* Convergence if Newton decrement < tol */
static int has_converged(NML_work *w)
{
    const double newton_dec =
        cblas_ddot(w->two_n_plus_one, w->grad, 1, w->neg_dir, 1);
    if (w->verbose)
    {
        w->grad_norm = cblas_dnrm2(w->two_n_plus_one, w->grad, 1);
        printf("grad_norm/obj: %.6e \t %.6f \n", w->grad_norm, w->obj);
    }

    return (newton_dec < w->tol);
}

/* Computes the step size.

    NOTE: When this function is called, w->neg_dir must contain the NEGATIVE
          search direction.
*/
static void compute_stepsize(NML_work *w)
{
    w->step_size = 1 / (w->beta); /* corresponds to initial step size 1 */
    int status, i;

    /* backtrack to ensure that the new iterate is in dom f */
    do
    {
        w->step_size *= w->beta;

        w->z[0] = 2 * (w->xy[0] - w->step_size * w->neg_dir[0]);
        for (i = 1; i < w->n_plus_one; i++)
        {
            w->z[i] = (w->xy[i] - w->step_size * w->neg_dir[i]) -
                      (w->xy[w->n + i] - w->step_size * w->neg_dir[w->n + i]) * I;
        }

        status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, w->n);

    } while (status == 1);

    lower_tri_diag_isqrt_mult(w->n_plus_one, w->sigma2, w->chol_toep);

    /* backtrack until descent condition is satisfied */
    const double dir_der = -cblas_ddot(w->two_n_plus_one, w->grad, 1, w->neg_dir, 1);

    while (1)
    {
        w->new_obj = compute_obj(w);

        if (w->new_obj < w->obj + w->alpha * w->step_size * dir_der)
        {
            break;
        }

        w->step_size *= w->beta;
        w->z[0] = 2 * (w->xy[0] - w->step_size * w->neg_dir[0]);
        for (i = 1; i < w->n_plus_one; i++)
        {
            w->z[i] = (w->xy[i] - w->step_size * w->neg_dir[i]) -
                      (w->xy[w->n + i] - w->step_size * w->neg_dir[w->n + i]) * I;
        }
        status = lev_dur_complex(w->z, w->chol_toep, w->sigma2, w->n);
        lower_tri_diag_isqrt_mult(w->n_plus_one, w->sigma2, w->chol_toep);
    }
    w->obj = w->new_obj;
}

void Newton(NML_work *w, NML_out *output)
{
    int i, k, info;
    int num_evals_found;
    int unused_ifail;
    double unused_eigvec;

    w->obj = compute_obj(w);

    for (i = 0; i < w->max_iter; i++)
    {
        compute_derivatives_packed(w);

        /* Cholesky factorization of Hessian, modify it if necessary */
        while (1)
        {
            memcpy(w->chol_hess_packed, w->hess_packed,
                   sizeof(double) * w->n_plus_one * w->two_n_plus_one);
            info = LAPACKE_dpptrf(LAPACK_COL_MAJOR, 'L', w->two_n_plus_one,
                                  w->chol_hess_packed);
            if (info == 0)
            {
                break;
            }

            if (w->verbose)
            {
                printf(
                    "iter %i. Hessian not PD. Adding multiple of the identity. \n ",
                    i);
            }

            output->num_of_hess_chol_fails += 1;

            /* compute eigenvalues of packed matrix (overwrites, so copy first) */
            memcpy(w->chol_hess_packed, w->hess_packed,
                   sizeof(double) * w->n_plus_one * w->two_n_plus_one);
            info = LAPACKE_dspevx(LAPACK_COL_MAJOR, 'N', 'A', 'L', w->two_n_plus_one,
                                  w->chol_hess_packed, 0.0, 0.0, 0.0, 0.0, -1.0,
                                  &num_evals_found, w->hess_evals, &unused_eigvec,
                                  1.0, &unused_ifail);

            w->hess_evals[0] = MIN(-0.1, w->hess_evals[0]) * 1.05;
            /* add multiple of the identity to Hessian */
            for (k = 0; k < w->two_n_plus_one; k++)
            {
                w->hess_packed[k * (w->two_n_plus_one + 1) - k * (k + 1) / 2] -=
                    w->hess_evals[0];
            }
        }

        /* compute Newton direction (negative direction stored in w->neg_dir) */
        memcpy(w->neg_dir, w->grad, sizeof(double) * w->two_n_plus_one);
        cblas_dtpsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    w->two_n_plus_one, w->chol_hess_packed, w->neg_dir, 1);
        cblas_dtpsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                    w->two_n_plus_one, w->chol_hess_packed, w->neg_dir, 1);

        if (has_converged(w))
        {
            break;
        }

        compute_stepsize(w);

        cblas_daxpy(w->two_n_plus_one, -w->step_size, w->neg_dir, 1, w->xy, 1);
    }

    output->obj = w->obj;
    output->grad_norm = w->grad_norm;
    output->iter = i;
    memcpy(output->x_sol, w->xy, sizeof(double) * (w->n + 1));
    memcpy(output->y_sol, w->xy + w->n + 1, sizeof(double) * w->n);
}
