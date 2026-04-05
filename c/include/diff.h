#ifndef NML_DIFF_H
#define NML_DIFF_H

#include "NML_solver_internal.h"

/* Computes the gradient and Hessian of f(x,y) = log det T(x,y) + Tr(T(x,y)^{-1} S)
   using FFT-accelerated evaluation. The gradient is stored in solver->work->grad
   and the Hessian is stored in packed lower-triangular format (column-major) in
   solver->work->hess_packed. */
void compute_derivatives_packed(NML_solver *solver);

#endif
