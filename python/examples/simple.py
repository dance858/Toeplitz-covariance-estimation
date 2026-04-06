"""Simple example: estimate a Toeplitz covariance matrix from data."""
import numpy as np
from scipy.linalg import toeplitz
from nml import NMLSolver


def random_toeplitz_cov(n):
    """Generate a random n x n positive definite Toeplitz matrix."""
    x = np.random.randn(n) + 1j * np.random.randn(n)
    M = 2 * n - 1
    z = np.fft.ifft(np.abs(np.fft.fft(x, M)) ** 2) / M
    first_col = z[:n]
    first_col[0] = first_col[0].real
    return toeplitz(first_col)

np.random.seed(0) 
# Ground truth: a 4x4 Toeplitz covariance matrix
n = 4
R_true = random_toeplitz_cov(n)

# Generate data
K = 5 * n
L = np.linalg.cholesky(R_true)
Z = L @ (np.random.randn(n, K) + 1j * np.random.randn(n, K)) / np.sqrt(2)

# Solve
solver = NMLSolver(n)
result = solver.solve(Z, verbose=True)
solver.free()

# Reconstruct the estimated Toeplitz matrix
x, y = result["x"], result["y"]
R_hat = toeplitz(np.concatenate([[2 * x[0]], x[1:] - 1j * y]))


# benchmark against sample covariance
sample_cov = (Z @ Z.conj().T) / K
error_sample_cov = np.linalg.norm(sample_cov - R_true, 'fro') / np.linalg.norm(R_true, 'fro')
error_nml = np.linalg.norm(R_hat - R_true, 'fro') / np.linalg.norm(R_true, 'fro')
print(f"Relative Frobenius norm error (sample covariance): {error_sample_cov:.4f}")
print(f"Relative Frobenius norm error (NML estimate): {error_nml:.4f}")
