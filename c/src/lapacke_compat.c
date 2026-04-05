#ifdef __APPLE__

#define ACCELERATE_NEW_LAPACK
#include "nml/platform/lapacke_compat.h"
#include <Accelerate/Accelerate.h>
#include <stdlib.h>

lapack_int LAPACKE_zpotrf(int matrix_layout, char uplo, lapack_int n,
                          lapack_complex_double *a, lapack_int lda)
{
    (void) matrix_layout;
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n;
    __LAPACK_int lda_ = lda;
    zpotrf_(&uplo, &n_, (__LAPACK_double_complex *) a, &lda_, &info);
    return (lapack_int) info;
}

lapack_int LAPACKE_dpptrf(int matrix_layout, char uplo, lapack_int n, double *ap)
{
    (void) matrix_layout;
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n;
    dpptrf_(&uplo, &n_, ap, &info);
    return (lapack_int) info;
}

lapack_int LAPACKE_dspevx(int matrix_layout, char jobz, char range, char uplo,
                          lapack_int n, double *ap, double vl, double vu,
                          lapack_int il, lapack_int iu, double abstol, lapack_int *m,
                          double *w, double *z, lapack_int ldz, lapack_int *ifail)
{
    (void) matrix_layout;
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n;
    __LAPACK_int il_ = il;
    __LAPACK_int iu_ = iu;
    __LAPACK_int ldz_ = ldz;
    __LAPACK_int m_ = 0;

    double *work = malloc(sizeof(double) * 8 * n);
    __LAPACK_int *iwork = malloc(sizeof(__LAPACK_int) * 5 * n);
    __LAPACK_int *ifail_ = malloc(sizeof(__LAPACK_int) * n);

    dspevx_(&jobz, &range, &uplo, &n_, ap, &vl, &vu, &il_, &iu_, &abstol, &m_, w, z,
            &ldz_, work, iwork, ifail_, &info);

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

#endif
