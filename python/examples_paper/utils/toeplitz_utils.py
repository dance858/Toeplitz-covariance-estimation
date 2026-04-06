"""Toeplitz matrix utilities."""
import numpy as np
from scipy.linalg import toeplitz


def random_toeplitz_cov(n):
    """Generate a random n x n positive definite Toeplitz matrix."""
    x = np.random.randn(n) + 1j * np.random.randn(n)
    M = 2 * n - 1
    z = np.fft.ifft(np.abs(np.fft.fft(x, M)) ** 2) / M
    first_col = z[:n]
    first_col[0] = first_col[0].real
    return toeplitz(first_col)


def generate_samples(T, N):
    """Generate N samples from a circular complex Gaussian with covariance T.

    Parameters
    ----------
    T : ndarray, shape (n, n)
        True covariance matrix (positive definite).
    N : int
        Number of samples.

    Returns
    -------
    X : ndarray, shape (n, N)
        Complex data matrix.
    """
    n = T.shape[0]
    L = np.linalg.cholesky(T)
    return L @ (np.random.randn(n, N) + 1j * np.random.randn(n, N)) / np.sqrt(2)
