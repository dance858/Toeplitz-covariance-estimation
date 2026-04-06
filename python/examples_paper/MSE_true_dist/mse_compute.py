"""Core MSE computation loop for Toeplitz covariance estimation."""
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.toeplitz_utils import generate_samples
from utils.estimates_cov import estimates_cov


def mse_compute(samples, true_cov, MC_runs):
    """Run Monte Carlo MSE evaluation over multiple sample sizes.

    Parameters
    ----------
    samples : array_like
        Sample sizes to evaluate.
    true_cov : ndarray, shape (n, n)
        True Toeplitz covariance matrix.
    MC_runs : int
        Number of Monte Carlo runs per sample size.

    Returns
    -------
    dict with keys:
        'MSE_NML', 'MSE_AML', 'MSE_DA' : ndarrays of MSE per sample size
        'NML_mean_time', 'AML_mean_time', 'DA_mean_time' : float
    """
    n = true_cov.shape[0]
    first_row_real = np.real(true_cov[0, :])
    first_row_imag = np.imag(true_cov[0, 1:])

    n_samples = len(samples)
    MSE_NML = np.zeros(n_samples)
    MSE_AML = np.zeros(n_samples)
    MSE_DA = np.zeros(n_samples)

    NML_total_time = 0.0
    AML_total_time = 0.0
    DA_total_time = 0.0
    aml_count = 0

    for ii, N in enumerate(samples):
        for run in range(MC_runs):
            X = generate_samples(true_cov, N)

            _, da_out, aml_out, nml_out = estimates_cov(X)

            NML_cov = nml_out["estimate"]
            DA_cov = da_out["estimate"]

            MSE_NML[ii] += (np.linalg.norm(first_row_real - np.real(NML_cov[0, :])) ** 2
                            + np.linalg.norm(first_row_imag - np.imag(NML_cov[0, 1:])) ** 2)
            MSE_DA[ii] += (np.linalg.norm(first_row_real - np.real(DA_cov[0, :])) ** 2
                           + np.linalg.norm(first_row_imag - np.imag(DA_cov[0, 1:])) ** 2)

            NML_total_time += nml_out["solve_time"]
            DA_total_time += da_out["solve_time"]

            if aml_out is not None:
                AML_cov = aml_out["estimate"]
                MSE_AML[ii] += (np.linalg.norm(first_row_real - np.real(AML_cov[0, :])) ** 2
                                + np.linalg.norm(first_row_imag - np.imag(AML_cov[0, 1:])) ** 2)
                AML_total_time += aml_out["solve_time"]
                aml_count += 1

        print(f"  N = {N} done")

    total_runs = n_samples * MC_runs
    MSE_NML /= MC_runs
    MSE_DA /= MC_runs
    MSE_AML /= MC_runs

    return {
        "MSE_NML": MSE_NML,
        "MSE_AML": MSE_AML,
        "MSE_DA": MSE_DA,
        "NML_mean_time": NML_total_time / total_runs,
        "AML_mean_time": AML_total_time / max(aml_count, 1),
        "DA_mean_time": DA_total_time / total_runs,
    }
