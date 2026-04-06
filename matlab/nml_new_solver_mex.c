#include "matrix.h"
#include "mex.h"
#include "nml/NML_solver.h"
#include <stdint.h>

/* nml_new_solver(n, tol, beta, alpha, max_iter) -> solver_ptr (uint64)
   n is the dimension of the covariance matrix. */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 5)
    {
        mexErrMsgTxt("Usage: solver = nml_new_solver(n, tol, beta, alpha, max_iter)");
        return;
    }

    int n = (int) mxGetScalar(prhs[0]);
    double tol = mxGetScalar(prhs[1]);
    double beta = mxGetScalar(prhs[2]);
    double alpha = mxGetScalar(prhs[3]);
    int max_iter = (int) mxGetScalar(prhs[4]);

    NML_solver *solver = nml_new_solver(n - 1, tol, beta, alpha, max_iter);
    if (!solver)
    {
        mexErrMsgTxt("Failed to create solver");
        return;
    }

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *) mxGetData(plhs[0])) = (uint64_t) solver;
}
