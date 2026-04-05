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
    L[1] = kappa;

    int k;
    for (k = 1; k < p; k++)
    {
        double *Lk = L + k * (k + 1) / 2;        /* column k in packed L */
        double *Lk1 = L + (k + 1) * (k + 2) / 2; /* column k+1 in packed L */

        kappa = -1 / sigma2[k] * (y[k + 1] + cblas_ddot(k, y + 1, 1, Lk, 1));
        if (fabs(kappa) > 1) break;
        sigma2[k + 1] = sigma2[k] * (1 - kappa * kappa);

        Lk1[0] = kappa;
        memcpy(Lk1 + 1, Lk, sizeof(double) * k);
        cblas_daxpy(k, kappa, Lk, -1, Lk1 + 1, 1);
        Lk1[1 + k] = 1;
    }

    if (k != p)
    {
        return 1;
    }
    else
    {
        reverse_real(L, (p + 1) * (p + 2) / 2);
        reverse_real(sigma2, p + 1);
        return 0;
    }
}

int lev_dur_complex(const nml_complex *y, nml_complex *L, double *sigma2, int p)
{
    /* initialization */
    nml_complex kappa = -y[1] / y[0]; /* reflection coefficient */
    if (cabs(kappa) > 1) return 1;
    sigma2[0] = creal(y[0]);
    sigma2[1] = creal(y[0]) - cabs(y[1]) * cabs(y[1]) / y[0];
    L[0] = L[2] = 1;
    L[1] = kappa;

    int k;
    for (k = 1; k < p; k++)
    {
        nml_complex *Lk = L + k * (k + 1) / 2;
        nml_complex *Lk1 = L + (k + 1) * (k + 2) / 2;

        cblas_zdotu_sub(k, y + 1, 1, Lk, 1, &kappa);
        kappa = -1 / sigma2[k] * (y[k + 1] + kappa);
        if (cabs(kappa) > 1) break;
        sigma2[k + 1] = sigma2[k] * (1 - cabs(kappa) * cabs(kappa));

        Lk1[0] = kappa;
        zaxpy_variant(k, kappa, Lk, Lk1 + 1);
        Lk1[1 + k] = 1;
    }

    if (k != p)
    {
        return 1;
    }
    else
    {
        reverse_complex(L, (p + 1) * (p + 2) / 2);
        reverse_real(sigma2, p + 1);
        return 0;
    }
}
