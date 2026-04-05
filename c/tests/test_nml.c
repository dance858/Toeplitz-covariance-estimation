#include "nml/NML_solver.h"
#include "nml/levinson_durbin.h"
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg)                                                           \
    do                                                                              \
    {                                                                               \
        if (!(cond))                                                                \
        {                                                                           \
            printf("  FAIL: %s\n", msg);                                            \
            return 1;                                                               \
        }                                                                           \
    } while (0)

#define RUN_TEST(fn)                                                                \
    do                                                                              \
    {                                                                               \
        tests_run++;                                                                \
        printf("Running %s ...\n", #fn);                                            \
        if (fn() == 0)                                                              \
        {                                                                           \
            printf("  PASS\n");                                                     \
            tests_passed++;                                                         \
        }                                                                           \
    } while (0)

/* Verify that T * T_inv ≈ I, where T_inv = L_scaled * L_scaled^T.
   L is in packed lower-triangular storage (column-major), dimension m = p+1.
   T_inv = L * diag(1/sigma2) * L^T, equivalently L_scaled * L_scaled^T
   where L_scaled has columns scaled by 1/sqrt(sigma2). */
static int check_cholesky_factor(const double *y, const double *L_packed,
                                 const double *sigma2, int p)
{
    int m = p + 1;
    double L_full[16] = {0}; /* max 4x4 */
    double T[16] = {0};
    double T_inv[16] = {0};

    /* Unpack L to full lower triangular and scale columns by 1/sqrt(sigma2) */
    int idx = 0;
    for (int col = 0; col < m; col++)
    {
        double scale = 1.0 / sqrt(sigma2[col]);
        for (int row = col; row < m; row++)
        {
            L_full[row + col * m] = L_packed[idx++] * scale;
        }
    }

    /* Build Toeplitz matrix T from first column y */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) T[i + j * m] = y[abs(i - j)];

    /* Compute T_inv = L_full * L_full^T */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            for (int k = 0; k < m; k++)
                T_inv[i + j * m] += L_full[i + k * m] * L_full[j + k * m];

    /* Check T * T_inv ≈ I */
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double val = 0;
            for (int k = 0; k < m; k++) val += T[i + k * m] * T_inv[k + j * m];
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(val - expected) > 1e-10) return 1;
        }
    }
    return 0;
}

/* --------------------------------------------------------------------------
   Test 1: Levinson-Durbin succeeds on a known PD Toeplitz matrix
   -------------------------------------------------------------------------- */
static int test_levinson_durbin_pd(void)
{
    /* First column of a 4x4 PD Toeplitz matrix: [2, 0.5, 0.25, 0.125] */
    int p = 3; /* dimension is p+1 = 4 */
    double y[] = {2.0, 0.5, 0.25, 0.125};
    double L[(3 + 1) * (3 + 2) / 2]; /* (p+1)*(p+2)/2 = 10 */
    double sigma2[4];                /* p+1 = 4 */

    int status = lev_dur_real(y, L, sigma2, p);
    ASSERT(status == 0, "lev_dur_real should return 0 for PD matrix");

    /* All sigma2 values must be positive for PD */
    for (int i = 0; i <= p; i++)
    {
        ASSERT(sigma2[i] > 0, "sigma2 values must be positive");
    }

    /* Verify T * T_inv ≈ I */
    ASSERT(check_cholesky_factor(y, L, sigma2, p) == 0,
           "T * T_inv should be close to identity");

    return 0;
}

/* --------------------------------------------------------------------------
   Test 2: Levinson-Durbin rejects non-PD input
   -------------------------------------------------------------------------- */
static int test_levinson_durbin_non_pd(void)
{
    /* [1, 2]: off-diagonal > diagonal, not PD */
    int p = 1;
    double y[] = {1.0, 2.0};
    double L[3];
    double sigma2[2];

    int status = lev_dur_real(y, L, sigma2, p);
    ASSERT(status == 1, "lev_dur_real should return 1 for non-PD matrix");

    return 0;
}

/* --------------------------------------------------------------------------
   Simple LCG random number generator (deterministic, no srand dependency)
   -------------------------------------------------------------------------- */
static unsigned long long rng_state = 12345;

static double rand_normal(void)
{
    /* Box-Muller transform using LCG */
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double) (rng_state >> 33) / (double) (1ULL << 31);
    if (u1 < 1e-15) u1 = 1e-15;
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double) (rng_state >> 33) / (double) (1ULL << 31);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/* --------------------------------------------------------------------------
   Test 3: Full NML solver end-to-end
   -------------------------------------------------------------------------- */
static int test_nml_solver(void)
{
    int n = 3;      /* covariance dimension is n+1 = 4 */
    int n1 = n + 1; /* = 4 */
    int K = 500;    /* number of samples */
    double rho = 0.5;

    /* Build true Toeplitz covariance: R[i][j] = rho^|i-j| */
    double R[4][4];
    for (int i = 0; i < n1; i++)
        for (int j = 0; j < n1; j++) R[i][j] = pow(rho, abs(i - j));

    /* Cholesky factor L (column-major, lower triangular)
       For this simple matrix, compute manually via the standard algorithm */
    double L[4][4];
    memset(L, 0, sizeof(L));
    for (int j = 0; j < n1; j++)
    {
        double sum = 0;
        for (int k = 0; k < j; k++) sum += L[j][k] * L[j][k];
        L[j][j] = sqrt(R[j][j] - sum);
        for (int i = j + 1; i < n1; i++)
        {
            sum = 0;
            for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];
            L[i][j] = (R[i][j] - sum) / L[j][j];
        }
    }

    /* Generate Z = L * noise, column-major, size n1 x K */
    nml_complex *Z = malloc(sizeof(nml_complex) * n1 * K);
    rng_state = 42; /* reset for reproducibility */
    for (int k = 0; k < K; k++)
    {
        double noise[4];
        for (int i = 0; i < n1; i++) noise[i] = rand_normal();
        for (int i = 0; i < n1; i++)
        {
            double val = 0;
            for (int j = 0; j <= i; j++) val += L[i][j] * noise[j];
            Z[i + k * n1] = val;
        }
    }

    /* Call solver */
    NML_solver *solver = nml_new_solver(n, 1e-8, 0.8, 0.05, 200);
    NML_result *result = nml_new_result(n);
    int ret = nml_solve(solver, Z, K, result, 1);
    ASSERT(ret == 0, "solve should return 0");
    ASSERT(result->iter < 50, "should converge in < 50 iterations");
    ASSERT(isfinite(result->obj), "objective should be finite");
    ASSERT(result->grad_norm < 1e-4, "gradient norm should be small at convergence");

    /* x[0] stores half the diagonal: should be close to 0.5 (= 1.0/2) */
    ASSERT(fabs(result->x[0] - 0.5) < 0.15,
           "x[0] should be close to 0.5 (half the true diagonal)");

    /* x[1] should be close to rho = 0.5 */
    ASSERT(fabs(result->x[1] - rho) < 0.15, "x[1] should be close to rho");

    nml_free_result(result);
    nml_free_solver(solver);
    free(Z);
    return 0;
}

/* -------------------------------------------------------------------------- */
int main(void)
{
    printf("NML Test Suite\n");
    printf("==============\n\n");

    RUN_TEST(test_levinson_durbin_pd);
    RUN_TEST(test_levinson_durbin_non_pd);
    RUN_TEST(test_nml_solver);

    printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
    fftw_cleanup();
    return (tests_passed == tests_run) ? 0 : 1;
}
