"""Averaging Along Diagonals (AAD) baseline estimator."""
import numpy as np
from scipy.linalg import toeplitz


def aad(S):
    """Estimate Toeplitz covariance by averaging diagonals of S.

    Parameters
    ----------
    S : ndarray, shape (n, n)
        Sample covariance matrix.

    Returns
    -------
    dict with keys 'estimate', 'x', 'y'.
    """
    n = S.shape[0]
    z = np.array([np.mean(np.diagonal(S, k)) for k in range(n)])
    x = np.real(z)
    y = np.imag(z[1:])
    estimate = toeplitz(z)
    return {"estimate": estimate, "x": x, "y": y}
