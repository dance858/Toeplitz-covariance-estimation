#include "nml/levinson_durbin.h"
#include "linalg.h"
#include "nml/platform/blas_lapack.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>

int lev_dur_real(const double *y, double *L, double *sigma2, int p)
{
    /* initialization */
    double kappa = -y[1] / y[0]; /* reflection coefficient */
    if (fabs(kappa) > 1) return 1;
    sigma2[0] = y[0];
    sigma2[1] = y[0] + y[1] * kappa;
    L[0] = L[2] = 1;
    L[1] = kappa; /* a_{11} */

    int k;
    for (k = 1; k < p; k++)
    {
        kappa = -1 / sigma2[k] *
                (y[k + 1] + cblas_ddot(k, y + 1, 1, L + k * (k + 1) / 2, 1));
        if (fabs(kappa) > 1) break;
        sigma2[k + 1] = sigma2[k] * (1 - kappa * kappa);

        L[(k + 1) * (k + 2) / 2] = kappa; /* a_{k+1, k+1} */
        cblas_dcopy(k, L + k * (k + 1) / 2, 1, L + (k + 1) * (k + 2) / 2 + 1, 1);
        cblas_daxpy(k, kappa, L + k * (k + 1) / 2, -1, L + (k + 1) * (k + 2) / 2 + 1,
                    1);
        L[(k + 1) * (k + 2) / 2 + 1 + k] = 1;
    }

    /* if the matrix is not PD, the absolute value check on kappa will be
       triggered */
    if (k != p)
    {
        return 1;
    }
    else
    {
        /* reverse memory */
        reverse_real(L, (p + 1) * (p + 2) / 2);
        reverse_real(sigma2, p + 1);
        return 0;
    }
}

int lev_dur_complex(const nml_complex *y, nml_complex *L, double *sigma2, int p)
{
    /* initialization */
    double complex kappa = -y[1] / y[0]; /* reflection coefficient */
    if (cabs(kappa) > 1) return 1;
    sigma2[0] = creal(y[0]); /* y[0] is always real but we do like this for safety */
    sigma2[1] = creal(y[0]) -
                cabs(y[1]) * cabs(y[1]) / y[0]; /* the inner argument is always real
                                                   but we do like this for safety */
    L[0] = L[2] = 1;
    L[1] = kappa; /* a_{11} */

    int k;
    for (k = 1; k < p; k++)
    {
        cblas_zdotu_sub(k, y + 1, 1, L + k * (k + 1) / 2, 1,
                        &kappa); /* kappa_k = -1/sigma2[k] * (y_{k+1} + y1 a_{kk} +
                                    y_2 a_{k, k-1} + ... + y_k a_{k, 1}) */
        kappa = -1 / sigma2[k] * (y[k + 1] + kappa);
        if (cabs(kappa) > 1) break;
        sigma2[k + 1] = sigma2[k] * (1 - cabs(kappa) * cabs(kappa));

        L[(k + 1) * (k + 2) / 2] = kappa; /* a_{k+1, k+1} */
        zaxpy_variant(k, &kappa, L + k * (k + 1) / 2, L + (k + 1) * (k + 2) / 2 + 1);
        L[(k + 1) * (k + 2) / 2 + 1 + k] = 1;
    }
    /* if the matrix is not PD, the absolute value check on kappa will be
       triggered */
    if (k != p)
    {
        return 1;
    }
    else
    {
        /* reverse memory */
        reverse_complex(L, (p + 1) * (p + 2) / 2);
        reverse_real(sigma2, p + 1);
        return 0;
    }
}
