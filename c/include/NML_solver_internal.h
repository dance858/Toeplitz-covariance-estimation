#ifndef NML_SOLVER_INTERNAL_H
#define NML_SOLVER_INTERNAL_H

#include "nml/NML_solver.h"
#include <complex.h>
#include <fftw3.h>

typedef struct
{
    double tol;
    double beta;
    double alpha;
    int max_iter;
} NML_settings;

typedef struct
{
    /* Iterate */
    double *xy;        /* [x, y]^T, size 2n + 1 */
    double complex *z; /* auxiliary complex variable, size n + 1 */

    /* Gradient and Hessian */
    double complex *grad_help; /* size N, used when computing gradient */
    double *grad;              /* [grad_x, grad_y], size 2n + 1 */
    double complex *F;         /* size N x N */
    double complex *G;         /* size N x N */
    double complex *hess_help; /* size N x N, used when computing Hessian */
    double *hess_packed;       /* packed lower triangular Hessian */
    double *chol_hess_packed;  /* packed Cholesky factor of Hessian */
    double *hess_evals;        /* eigenvalues of Hessian, size 2n + 1 */

    /* Toeplitz factorization */
    double complex *chol_toep;      /* packed Cholesky factor of T^{-1} */
    double complex *full_chol_toep; /* full Cholesky factor of T^{-1} */
    double *sigma2;                 /* Levinson-Durbin diagonal */
    double complex *L_full;         /* Cholesky factor of sample covariance */
    double complex *RHL;            /* chol_toep^H * L product */
    double complex *sample_cov;     /* sample covariance matrix */

    /* Search direction */
    double *neg_dir; /* negative search direction, size 2n + 1 */

    /* DFTs */
    double complex *R_DFT;
    double complex *A_DFT;

    /* FFTW plans */
    fftw_plan plan_R_DFT;
    fftw_plan plan_A_DFT;
    fftw_plan plan_grad_help;
    fftw_plan plan_hess_help;

    /* Per-iteration scalars */
    double step_size;
    double obj;
    double new_obj;
    double grad_norm;
} NML_workspace;

struct NML_solver
{
    int n;
    int n_plus_one;
    int two_n_plus_one;
    int N; /* FFT dimension = 2*(n+1) */

    NML_settings settings;
    NML_workspace work;

    /* Per-solve state */
    int K;
    int verbose;
};

#endif
