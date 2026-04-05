#include "matrix.h"
#include "mex.h"
#include "nml/NML_solver.h"
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* real_Z, imag_Z, n, K, verbose, tol, beta, alpha, max_iter  */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    /* check that the number of arguments is correct */
    if (nrhs > 9)
    {
        mexErrMsgTxt("Too many input arguments");
        return;
    }
    else if (nrhs < 9)
    {
        mexErrMsgTxt("Too few input arguments");
        return;
    }

    /* parse input*/
    int n = (int) mxGetScalar(prhs[2]);
    double K = mxGetScalar(prhs[3]);
    int verbose = (int) mxGetScalar(prhs[4]);
    double tol = mxGetScalar(prhs[5]);
    double beta = mxGetScalar(prhs[6]);
    double alpha = mxGetScalar(prhs[7]);
    int max_iter = (int) mxGetScalar(prhs[8]);

    if (mxGetM(prhs[0]) == (n + 1) * K && mxGetM(prhs[1]) == (n + 1) * K)
    {

        /* real and imaginary parts of S*/
        double *real_Z;
        double *imag_Z;

        real_Z = mxGetPr(prhs[0]);
        imag_Z = mxGetPr(prhs[1]);

        double complex *Z;
        Z = malloc(sizeof(double complex) * (n + 1) * K);
        for (int i = 0; i < (n + 1) * K; i++)
        {
            Z[i] = real_Z[i] + I * imag_Z[i];
        }

        /* Call solver */
        NML_solver *solver = nml_new_solver(n, tol, beta, alpha, max_iter);
        NML_result *result = nml_new_result(n);
        int res = nml_solve(solver, Z, K, result, verbose);
        if (res != 0)
        {
            mexErrMsgTxt("Return error. Please report this to the developer.");
            return;
        }

        double *ptr;
        /* x */
        plhs[0] = mxCreateNumericMatrix(n + 1, 1, mxDOUBLE_CLASS, mxREAL);
        ptr = mxGetPr(plhs[0]);
        memcpy(ptr, result->x, sizeof(double) * (n + 1));

        /* y */
        plhs[1] = mxCreateNumericMatrix(n, 1, mxDOUBLE_CLASS, mxREAL);
        ptr = mxGetPr(plhs[1]);
        memcpy(ptr, result->y, sizeof(double) * n);

        /* gradient norm */
        plhs[2] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
        ptr = mxGetPr(plhs[2]);
        *ptr = result->grad_norm;

        /* objective value */
        plhs[3] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
        ptr = mxGetPr(plhs[3]);
        *ptr = result->obj;

        /* solve time */
        plhs[4] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
        ptr = mxGetPr(plhs[4]);
        *ptr = result->solve_time;

        /* number of iterations */
        plhs[5] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
        ptr = mxGetPr(plhs[5]);
        *ptr = result->iter;

        /* diagonal initialization succeeded */
        plhs[6] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
        ptr = mxGetPr(plhs[6]);
        *ptr = result->diag_init_succeeded;

        /* number of times Hessian was modified */
        plhs[7] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
        ptr = mxGetPr(plhs[7]);
        *ptr = result->num_of_hess_chol_fails;

        /* deallocate */
        free(Z);
        nml_free_result(result);
        nml_free_solver(solver);
    }
    else
    {
        mexErrMsgTxt("Incompatible dimensions. ");
    }
}
