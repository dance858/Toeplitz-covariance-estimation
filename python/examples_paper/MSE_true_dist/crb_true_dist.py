"""Fisher Information Matrix for Toeplitz covariance estimation."""
import numpy as np


def crb_true_dist(true_cov, N, n):
    """Compute the FIM for Toeplitz-parameterized covariance estimation.

    Parameters
    ----------
    true_cov : ndarray, shape (n, n)
        True Toeplitz covariance matrix.
    N : int
        Number of snapshots.
    n : int
        Dimension of the covariance matrix.

    Returns
    -------
    FIM : ndarray, shape (2*n-1, 2*n-1)
    """
    dim = 2 * n - 1
    FIM = np.zeros((dim, dim))
    R_inv = np.linalg.inv(true_cov)

    partials = []
    for i in range(dim):
        if i == 0:
            dT = np.eye(n)
        elif 1 <= i <= n - 1:
            E = np.diag(np.ones(n - i), -i)
            dT = E.T + E
        else:
            k = i - (n - 1)
            E = np.diag(np.ones(n - k), -k)
            dT = 1j * (E.T - E)
        partials.append(dT)

    for i in range(dim):
        Ri = R_inv @ partials[i]
        for j in range(i, dim):
            Rj = R_inv @ partials[j]
            FIM[i, j] = N * np.real(np.trace(Ri @ Rj))
            FIM[j, i] = FIM[i, j]

    return FIM
