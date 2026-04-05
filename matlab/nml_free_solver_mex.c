#include "matrix.h"
#include "mex.h"
#include "nml/NML_solver.h"
#include <stdint.h>

/* nml_free_solver(solver_ptr) */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
    {
        mexErrMsgTxt("Usage: nml_free_solver(solver)");
        return;
    }

    NML_solver *solver =
        (NML_solver *) (*((uint64_t *) mxGetData(prhs[0])));
    if (solver)
    {
        nml_free_solver(solver);
    }
}
