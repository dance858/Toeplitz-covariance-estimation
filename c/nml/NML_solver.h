#ifndef NML_SOLVER_H
#define NML_SOLVER_H

#include <complex.h>

typedef struct NML_solver NML_solver;

typedef struct
{
    double *x;                  /* real part of first column, size n+1 */
    double *y;                  /* imaginary part of first column, size n */
    double grad_norm;           /* norm of gradient at solution */
    double obj;                 /* objective value at solution */
    double solve_time;          /* solve time in seconds */
    int iter;                   /* number of Newton iterations */
    int diag_init_succeeded;    /* 1 if diagonal initialization succeeded */
    int num_of_hess_chol_fails; /* number of times the Hessian was modified */
} NML_result;

NML_solver *nml_new_solver(int n, double tol, double beta, double alpha, int max_iter);
NML_result *nml_new_result(int n);
int nml_solve(NML_solver *solver, const double complex *Z, int K,
              NML_result *result, int verbose);
void nml_free_solver(NML_solver *solver);
void nml_free_result(NML_result *result);

#endif
