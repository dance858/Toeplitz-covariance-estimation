"""Asymptotic Maximum Likelihood (AML) Toeplitz estimator."""
import time
import numpy as np
from scipy.linalg import toeplitz


def create_psi_aml(M):
    """Build the AML constraint matrix Psi.

    Parameters
    ----------
    M : int
        Dimension of the covariance matrix.

    Returns
    -------
    Psi : ndarray, shape (M*M, 2*M-1)
    """
    K = 2 * M - 1
    Omega = np.zeros((K, K), dtype=complex)
    Omega[0, 0] = 1
    col_counter = 1
    for row in range(1, K):
        if row % 2 == 1:  # even row in 1-based (row+1 is even)
            Omega[row, col_counter] = 1
            Omega[row, col_counter + 1] = 1j
        else:  # odd row in 1-based
            Omega[row, col_counter] = 1
            Omega[row, col_counter + 1] = -1j
            col_counter += 2

    Sigma = np.zeros((M * M, K), dtype=complex)
    IM = np.eye(M)
    Sigma[:, 0] = IM.flatten("F")
    col_counter = 1
    for m in range(1, M):
        Q_col = np.zeros((M, M))
        Q_col[:M - m, m:] = np.eye(M - m)
        Sigma[:, col_counter] = Q_col.flatten("F")
        Sigma[:, col_counter + 1] = Q_col.T.flatten("F")
        col_counter += 2

    return Sigma @ Omega


def aml(R_tilde, N, Psi):
    """Asymptotic Maximum Likelihood Toeplitz estimator.

    Parameters
    ----------
    R_tilde : ndarray, shape (M, M)
        Sample covariance matrix.
    N : int
        Number of data samples.
    Psi : ndarray, shape (M*M, 2*M-1)
        Constraint matrix from create_psi_aml.

    Returns
    -------
    dict with keys 'estimate', 'x', 'y', 'solve_time'.
    """
    t0 = time.perf_counter()
    M = R_tilde.shape[0]
    r_tilde = R_tilde.flatten("F")
    C_tilde = (1 / N) * np.kron(R_tilde.T, R_tilde)
    L = np.linalg.cholesky(C_tilde)
    Linv_Psi = np.linalg.solve(L, Psi)
    Linv_r = np.linalg.solve(L, r_tilde)
    phi_tilde = np.linalg.solve(Linv_Psi.conj().T @ Linv_Psi,
                                Linv_Psi.conj().T @ Linv_r)
    phi_tilde = np.real(phi_tilde)
    x = np.concatenate([[phi_tilde[0]], phi_tilde[1::2][:M - 1]])
    y = phi_tilde[2::2][:M - 1]
    T = toeplitz(x + 1j * np.concatenate([[0], y]))
    solve_time = time.perf_counter() - t0
    return {"estimate": T, "x": x, "y": y, "solve_time": solve_time}
