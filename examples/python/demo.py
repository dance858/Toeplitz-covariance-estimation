"""Demo: Toeplitz covariance estimation with NML.

Generates synthetic data from a Toeplitz covariance matrix and recovers
the covariance using the NML solver.
"""
import numpy as np
from nml import solve


def make_toeplitz_cov(rho, n):
    """Create a Toeplitz covariance matrix with exponential correlation."""
    row = np.array([rho**k for k in range(n)])
    from scipy.linalg import toeplitz
    return toeplitz(row)


def main():
    np.random.seed(42)
    n_plus_one = 8   # dimension
    K = 200           # number of samples
    rho = 0.5 + 0.3j  # correlation parameter

    # True Toeplitz first column
    true_col = np.array([rho**k for k in range(n_plus_one)])

    # Build true covariance and generate data
    from scipy.linalg import toeplitz, sqrtm
    R_true = toeplitz(true_col)
    L = np.linalg.cholesky(R_true)
    noise = (np.random.randn(n_plus_one, K) +
             1j * np.random.randn(n_plus_one, K)) / np.sqrt(2)
    Z = L @ noise

    # Solve
    result = solve(Z, verbose=True)

    print(f"\nSolver converged in {result['iter']} iterations")
    print(f"Objective: {result['obj']:.6f}")
    print(f"Time: {result['time']:.4f} s")

    # Recovered first column
    r_hat = result["x"] - 1j * np.concatenate([[0], result["y"]])
    r_hat[0] *= 2  # x[0] stores half the diagonal
    print(f"\nTrue first column:      {true_col[:4].real.round(3)}")
    print(f"Recovered first column: {r_hat[:4].real.round(3)}")


if __name__ == "__main__":
    main()
