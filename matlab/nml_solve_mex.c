#include "matrix.h"
#include "mex.h"
#include "nml/NML_solver.h"
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* [x, y, grad_norm, obj, solve_time, iter, diag_init_succeeded,
    num_hess_chol_fails] = nml_solve(solver_ptr, Z, verbose) */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3)
    {
        mexErrMsgTxt("Usage: [x, y, ...] = nml_solve(solver, Z, verbose)");
        return;
    }

    /* Recover solver pointer */
    NML_solver *solver =
        (NML_solver *) (*((uint64_t *) mxGetData(prhs[0])));
    if (!solver)
    {
        mexErrMsgTxt("Invalid solver handle");
        return;
    }

    /* Parse Z (complex matrix of size (n+1) x K) */
    mwSize rows = mxGetM(prhs[1]);
    mwSize cols = mxGetN(prhs[1]);
    int n = (int) rows - 1;
    int K = (int) cols;
    int verbose = (int) mxGetScalar(prhs[2]);

    double *real_Z = mxGetPr(prhs[1]);
    double *imag_Z = mxGetPi(prhs[1]);

    double complex *Z = malloc(sizeof(double complex) * rows * K);
    for (int i = 0; i < (int) (rows * K); i++)
    {
        double re = real_Z[i];
        double im = imag_Z ? imag_Z[i] : 0.0;
        Z[i] = re + I * im;
    }

    /* Solve */
    NML_result *result = nml_new_result(n);
    int ret = nml_solve(solver, Z, K, result, verbose);
    free(Z);

    if (ret != 0)
    {
        nml_free_result(result);
        mexErrMsgTxt("Solver returned error. Please report this to the developer.");
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

    /* grad_norm */
    plhs[2] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    *mxGetPr(plhs[2]) = result->grad_norm;

    /* obj */
    plhs[3] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    *mxGetPr(plhs[3]) = result->obj;

    /* solve_time */
    plhs[4] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    *mxGetPr(plhs[4]) = result->solve_time;

    /* iter */
    plhs[5] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    *mxGetPr(plhs[5]) = result->iter;

    /* diag_init_succeeded */
    plhs[6] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    *mxGetPr(plhs[6]) = result->diag_init_succeeded;

    /* num_of_hess_chol_fails */
    plhs[7] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    *mxGetPr(plhs[7]) = result->num_of_hess_chol_fails;

    nml_free_result(result);
}
