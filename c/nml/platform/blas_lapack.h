#ifndef NML_BLAS_LAPACK_H
#define NML_BLAS_LAPACK_H

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#ifdef NML_HAS_LAPACKE
#include <lapacke.h>
#else
#include "lapacke_compat.h"
#endif

#endif
