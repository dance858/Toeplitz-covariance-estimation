#ifndef NML_WORK
#define NML_WORK

#include <complex.h>
#include <fftw3.h>

typedef struct{
    int n;                            /* dimension */
    int n_plus_one;
    int two_n_plus_one;
    int N;                            /* dimension of FFT */
    int K;                            /* number of data points */
    double *xy;                       /* [x, y]^T, size 2n + 1 */
    double complex *z;                /* auxiliary complex variable, size n + 1 */

    /* Algorithmic parameters */
    double tol;                       /* termination criteria */
    double beta;                      /* linesearch parameter */
    double alpha;                     /* linesearch parameter */
    int verbose;
    double tau;                       /* factor to increase diagonal entries of Hessian if Cholesky fails */
    int max_iter;
  
    double complex *grad_help;        /* size N x 1, used when computing the gradient */
    double *grad;                     /* grad = [grad_x, grad_y], size 2n + 1 */
    double complex *F;                /* size N x N */
    double complex *G;                /* size N x N */
    double complex *hess_help;        /* size N x N, used when computing the Hessian */
    //double *hess;                   /* hess = [hess_xx, hess_yx^H; hess_yx, hess_yy], size (2n+1) x (2n+1) */
    double *hess_packed;              /* packed storage (lower triangular part, column-major order) of hess, size (2n+1)*(2n+2)/2 */                     
    double *chol_hess_packed;         /* packed storage (lower triangular part, column-major order) of lower Cholesky factor of Hessian */
    double *hess_evals;               /* eigenvalues of Hessian, array of size (2n + 1) */

    double complex *chol_toep;        /* packed storage (column-major order) of lower Cholesky factor of T(x, y)^{-1} */ 
    double complex *full_chol_toep;   /* full storage (column-major order) of lower Cholesky factor of T(x, y)^{-1} */                        
    double *sigma2;                   /* required for the Levinson-Durbin algorithm */
    double complex *L_full;           /* full storage (column-major order) of lower Cholesky factor of sample covariance */
    double complex *RHL;              /* represents chol_toep^H*L and chol_toep*(chol_toep^H*L) */ 
    double complex *sample_cov;       /* sample covariance matrix */
    
    double *neg_dir;                  /* negative search direction (both x and y), size 2n x 1*/

    double step_size;
    double obj;                       /* objective value in current iterate */
    double new_obj;                   /* objective value in candidate iterate */
    double grad_norm;

    /* DFTs of R and A */
    double complex *R_DFT;    
    double complex *A_DFT;

    /* Plans for FFTW */
    fftw_plan plan_R_DFT;
    fftw_plan plan_A_DFT;
    fftw_plan plan_grad_help;
    fftw_plan plan_hess_help;
} NML_work;

typedef struct{
   double grad_norm;                        /* norm of gradient, last iterate */
   double obj;                              /* objective value, last iterate */
   int iter;                                /* number of iterations until convergence */
   int diag_init_succeded;                  /* 1 if the diagonal initialization yielded a valid initial point, otherwise zero */
   int num_of_hess_chol_fails;              /* number of times the Hessian was modified */
   double total_time;                       /* in seconds*/
   double *x_sol;                           
   double *y_sol;
} NML_out;

#endif
