/* LAPACKE shim: provides the 3 LAPACKE functions used by NML by calling
   the Fortran-style LAPACK routines available in Accelerate (macOS) or
   OpenBLAS (Windows/other). Only compiled when system LAPACKE is not found. */

#include "nml/platform/lapacke_compat.h"
#include <stdlib.h>

/* Fortran name mangling: trailing underscore (gfortran/Accelerate convention) */
#define LAPACK_GLOBAL(name) name##_

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
typedef __LAPACK_int LAPACK_INT;
typedef __LAPACK_double_complex LAPACK_ZTYPE;
#else
typedef int LAPACK_INT;
typedef double _Complex LAPACK_ZTYPE;
extern void LAPACK_GLOBAL(zpotrf)(char *uplo, LAPACK_INT *n, LAPACK_ZTYPE *a,
                                  LAPACK_INT *lda, LAPACK_INT *info);
extern void LAPACK_GLOBAL(dpptrf)(char *uplo, LAPACK_INT *n, double *ap,
                                  LAPACK_INT *info);
extern void LAPACK_GLOBAL(dspevx)(char *jobz, char *range, char *uplo, LAPACK_INT *n,
                                  double *ap, double *vl, double *vu, LAPACK_INT *il,
                                  LAPACK_INT *iu, double *abstol, LAPACK_INT *m,
                                  double *w, double *z, LAPACK_INT *ldz,
                                  double *work, LAPACK_INT *iwork, LAPACK_INT *ifail,
                                  LAPACK_INT *info);
#endif

lapack_int LAPACKE_zpotrf(int matrix_layout, char uplo, lapack_int n,
                          lapack_complex_double *a, lapack_int lda)
{
    (void) matrix_layout;
    LAPACK_INT info = 0;
    LAPACK_INT n_ = n;
    LAPACK_INT lda_ = lda;
    LAPACK_GLOBAL(zpotrf)(&uplo, &n_, (LAPACK_ZTYPE *) a, &lda_, &info);
    return (lapack_int) info;
}

lapack_int LAPACKE_dpptrf(int matrix_layout, char uplo, lapack_int n, double *ap)
{
    (void) matrix_layout;
    LAPACK_INT info = 0;
    LAPACK_INT n_ = n;
    LAPACK_GLOBAL(dpptrf)(&uplo, &n_, ap, &info);
    return (lapack_int) info;
}

lapack_int LAPACKE_dspevx(int matrix_layout, char jobz, char range, char uplo,
                          lapack_int n, double *ap, double vl, double vu,
                          lapack_int il, lapack_int iu, double abstol, lapack_int *m,
                          double *w, double *z, lapack_int ldz, lapack_int *ifail)
{
    (void) matrix_layout;
    LAPACK_INT info = 0;
    LAPACK_INT n_ = n;
    LAPACK_INT il_ = il;
    LAPACK_INT iu_ = iu;
    LAPACK_INT ldz_ = ldz;
    LAPACK_INT m_ = 0;

    double *work = malloc(sizeof(double) * 8 * n);
    LAPACK_INT *iwork = malloc(sizeof(LAPACK_INT) * 5 * n);
    LAPACK_INT *ifail_ = malloc(sizeof(LAPACK_INT) * n);

    LAPACK_GLOBAL(dspevx)
    (&jobz, &range, &uplo, &n_, ap, &vl, &vu, &il_, &iu_, &abstol, &m_, w, z, &ldz_,
     work, iwork, ifail_, &info);

    *m = (lapack_int) m_;
    if (ifail != NULL)
    {
        for (int i = 0; i < (int) m_; i++) ifail[i] = (lapack_int) ifail_[i];
    }

    free(work);
    free(iwork);
    free(ifail_);
    return (lapack_int) info;
}
