#ifndef NML_BLAS_LAPACK_H
#define NML_BLAS_LAPACK_H

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include "lapacke_compat.h"
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#endif
