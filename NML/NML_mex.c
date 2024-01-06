#include "mex.h"
#include "matrix.h"
#include "NML_solve.h"
#include "NML_work.h"
#include <complex.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* real_Z, imag_Z, n, K, verbose, tol, beta, alpha, max_iter  */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    /* check that the number of arguments is correct */
    if (nrhs > 9){
        mexErrMsgTxt("Too many input arguments");
        return;
    }
    else if (nrhs < 9){
        mexErrMsgTxt("Too few input arguments");
        return;
    }
   
    /* parse input*/
    int n = (int)mxGetScalar(prhs[2]);
    double K = mxGetScalar(prhs[3]);
    int verbose = (int)mxGetScalar(prhs[4]);
    double tol = mxGetScalar(prhs[5]);
    double beta = mxGetScalar(prhs[6]);
    double alpha = mxGetScalar(prhs[7]);
    int max_iter = (int)mxGetScalar(prhs[8]);
       
    if (mxGetM(prhs[0]) == (n + 1) * K && 
        mxGetM(prhs[1]) == (n + 1) * K){
        
        /* real and imaginary parts of S*/
        double *real_Z;
        double *imag_Z;

        real_Z = mxGetPr(prhs[0]);
        imag_Z = mxGetPr(prhs[1]);

        double complex *Z;
        Z = malloc(sizeof(double complex) * (n + 1) * K);
        for (int i = 0; i < (n + 1) * K ; i++){
            Z[i] = real_Z[i] + I*imag_Z[i];
        }

        /* ------------------------------------------- 
                 Call NML implemented in c                   
           ------------------------------------------- */
          
          /* prepare output */
          NML_out *output = malloc(sizeof(*output));
          int res = NML(Z, n, K, output, tol, beta, alpha, verbose, max_iter);
          if (res != 0) {
              mexErrMsgTxt("Return error. Please report this to the developer.");
              return;
          }
          
          double *ptr;
          /*output x_sol */
          plhs[0] = mxCreateNumericMatrix(n + 1, 1, mxDOUBLE_CLASS, mxREAL);
          ptr = mxGetPr(plhs[0]);
          memcpy(ptr, output->x_sol, sizeof(double) * (n + 1));
          
          /* output y_sol */
          plhs[1] = mxCreateNumericMatrix(n, 1, mxDOUBLE_CLASS, mxREAL);
          ptr = mxGetPr(plhs[1]);
          memcpy(ptr, output->y_sol, sizeof(double) * n);

          /* output gradient norm  */
          plhs[2] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
          ptr = mxGetPr(plhs[2]);
          *ptr = output->grad_norm;

          /* output objective value */
          plhs[3] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
          ptr = mxGetPr(plhs[3]);
          *ptr = output->obj;

          /* output total time */
          plhs[4] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
          ptr = mxGetPr(plhs[4]);
          *ptr = output->total_time;

          /* output number of iterations */
          plhs[5] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
          ptr = mxGetPr(plhs[5]);
          *ptr = output->iter;

          /* output if diagonal initilization succeeded */
          plhs[6] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
          ptr = mxGetPr(plhs[6]);
          *ptr = output->diag_init_succeded;

          /* output number of times Hessian was modified */
          plhs[7] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
          ptr = mxGetPr(plhs[7]);
          *ptr = output->num_of_hess_chol_fails;  

          /* deallocate data matrix and output memory that is allocated inside NML */
          free(Z);
          free(output->x_sol);
          free(output->y_sol);
          free(output);
    }
    else{
        mexErrMsgTxt("Incompatible dimensions. ");
    }
}
