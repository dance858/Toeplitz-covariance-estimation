#ifndef NML_TYPES_H
#define NML_TYPES_H

#include <complex.h>
typedef double _Complex nml_complex;

/* On Windows with clang-cl, the UCRT <complex.h> defines creal/cimag/cabs/conj
   for _Dcomplex (MSVC's struct type), not _Complex double. Override with Clang
   builtins which work with C99 _Complex double. */
#if defined(_WIN32) && defined(__clang__)
#undef creal
#undef cimag
#undef cabs
#undef conj
#define creal(z) __real__(z)
#define cimag(z) __imag__(z)
#define cabs(z) __builtin_cabs(z)
#define conj(z) __builtin_conj(z)
#endif

/* Platform-independent double complex imaginary unit. Avoids the Windows
   issue where I is _Fcomplex (float). */
#define NML_I ((nml_complex){0.0, 1.0})

/* FFTW on Windows defines fftw_complex as double[2], not _Complex double.
   They are layout-compatible, so a cast is safe. On non-Windows platforms
   where fftw_complex is _Complex double, this is a no-op cast. */
#define NML_FFTW(ptr) ((fftw_complex *) (ptr))

#endif
