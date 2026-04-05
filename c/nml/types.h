#ifndef NML_TYPES_H
#define NML_TYPES_H

#ifdef _MSC_VER
#include <complex.h>
typedef _Dcomplex nml_complex;
#else
#include <complex.h>
typedef double complex nml_complex;
#endif

#endif
