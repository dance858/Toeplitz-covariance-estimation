"""DOA estimation utilities for the NML demo.

Provides ULA data generation, Root-MUSIC estimation, and Cramér-Rao bounds.
Based on https://github.com/dance858/T-Rex-EM/tree/main/experiments/DOA.
"""
import numpy as np


def generate_ula_data(power_source, sig2, d, m, K, wavelength, theta_rad):
    """Generate synthetic ULA data with uniform Gaussian noise.

    Parameters
    ----------
    power_source : float
        Source signal power.
    sig2 : float
        Noise variance.
    d : float
        Sensor spacing in wavelengths.
    m : int
        Number of sensors.
    K : int
        Number of snapshots.
    wavelength : float
        Carrier wavelength.
    theta_rad : array_like
        Source angles in radians.

    Returns
    -------
    Y : ndarray, shape (m, K)
        Received signal matrix.
    true_cov : ndarray, shape (m, m)
        True covariance matrix.
    """
    theta_rad = np.asarray(theta_rad)
    n_sources = len(theta_rad)
    A = np.exp(-2j * np.pi / wavelength
               * d * np.arange(m).reshape(-1, 1) * np.sin(theta_rad))
    s = np.sqrt(power_source / 2) * (
        np.random.randn(n_sources, K) + 1j * np.random.randn(n_sources, K))
    e = np.sqrt(sig2 / 2) * (
        np.random.randn(m, K) + 1j * np.random.randn(m, K))
    Y = A @ s + e
    P = power_source * np.eye(n_sources)
    true_cov = A @ P @ A.conj().T + sig2 * np.eye(m)
    return Y, true_cov


def root_music(R, k, d, wavelength):
    """Root-MUSIC DOA estimator for uniform linear arrays.

    Parameters
    ----------
    R : ndarray, shape (m, m)
        Covariance matrix estimate.
    k : int
        Number of sources.
    d : float
        Sensor spacing in wavelengths.
    wavelength : float
        Carrier wavelength.

    Returns
    -------
    doa : ndarray, shape (k,)
        Estimated DOAs in radians, sorted.
    """
    m = R.shape[0]
    _, E = np.linalg.eigh(R)
    En = E[:, :-k]
    C = En @ En.conj().T

    # Build polynomial coefficients from anti-diagonals of C
    coeff = np.zeros(m - 1, dtype=np.complex128)
    for i in range(1, m):
        coeff[i - 1] = np.sum(np.diag(C, i))
    coeff = np.hstack((coeff[::-1], np.sum(np.diag(C)), coeff.conj()))

    z = np.roots(coeff)

    # Keep roots inside the unit circle
    mask = np.abs(z) < 1.0
    z = z[mask]

    # Select k roots closest to the unit circle
    sorted_indices = np.argsort(1.0 - np.abs(z))
    z = z[sorted_indices[:k]]

    # Convert roots to angles
    c = 2 * np.pi * d / wavelength
    sin_vals = np.angle(z) / c
    doa = np.sort(np.arcsin(np.clip(sin_vals, -1, 1)))
    return doa


def u_crb(P, theta_rad, sig2, m, d, N):
    """Unconditional (stochastic) Cramér-Rao bound for DOA estimation.

    Parameters
    ----------
    P : ndarray, shape (k, k)
        Source covariance matrix.
    theta_rad : array_like
        True DOAs in radians.
    sig2 : float
        Noise variance.
    m : int
        Number of sensors.
    d : float
        Sensor spacing in wavelengths.
    N : int
        Number of snapshots.

    Returns
    -------
    CRB : ndarray, shape (k, k)
        Cramér-Rao bound matrix for DOA parameters.
    """
    theta_rad = np.asarray(theta_rad)
    sensor_idx = np.arange(m).reshape(-1, 1)
    A = np.exp(-2j * np.pi * d * sensor_idx * np.sin(theta_rad))
    D = (-2j * np.pi * d * sensor_idx * np.cos(theta_rad)) * A
    R = A @ P @ A.conj().T + sig2 * np.eye(m)
    proj = np.eye(m) - A @ np.linalg.solve(A.conj().T @ A, A.conj().T)
    CRB = sig2 / (2 * N) * np.linalg.inv(
        np.real((D.conj().T @ proj @ D)
                * (P @ A.conj().T @ np.linalg.solve(R, A) @ P).T))
    return CRB


def s_crb(P, theta_rad, sig2, m, d, N):
    """Conditional (deterministic) Cramér-Rao bound for DOA estimation.

    Parameters
    ----------
    P : ndarray, shape (k, k)
        Source covariance matrix.
    theta_rad : array_like
        True DOAs in radians.
    sig2 : float
        Noise variance.
    m : int
        Number of sensors.
    d : float
        Sensor spacing in wavelengths.
    N : int
        Number of snapshots.

    Returns
    -------
    CRB : ndarray, shape (k, k)
        Cramér-Rao bound matrix for DOA parameters.
    """
    theta_rad = np.asarray(theta_rad)
    k = len(theta_rad)
    p = np.diag(P)
    sensor_idx = np.arange(m).reshape(-1, 1)
    A = np.exp(-2j * np.pi * d * sensor_idx * np.sin(theta_rad))
    DA = (-2j * np.pi * d * sensor_idx * np.cos(theta_rad)) * A
    R = A * p.reshape(1, -1) @ A.conj().T + sig2 * np.eye(m)
    R_inv = np.linalg.inv(R)
    R_inv = 0.5 * (R_inv + R_inv.conj().T)

    DRD = DA.conj().T @ R_inv @ DA
    DRA = DA.conj().T @ R_inv @ A
    ARD = A.conj().T @ R_inv @ DA
    ARA = A.conj().T @ R_inv @ A
    PP = np.outer(p, p)
    R_inv2 = R_inv @ R_inv

    FIM_tt = 2 * np.real((DRD.T * ARA + DRA.conj() * ARD) * PP)
    FIM_pp = np.real(ARA.T * ARA)
    FIM_ss = np.real(np.sum(np.diag(R_inv2)))
    FIM_tp = 2 * np.real(DRA.conj() * (p.reshape(-1, 1) * ARA))
    FIM_ts = 2 * np.real(
        p * np.sum(DA.conj() * (R_inv2 @ A), axis=0)).reshape(-1, 1)
    FIM_ps = np.real(
        np.sum(A.conj() * (R_inv2 @ A), axis=0)).reshape(-1, 1)

    FIM = np.block([
        [FIM_tt, FIM_tp, FIM_ts],
        [FIM_tp.T, FIM_pp, FIM_ps],
        [FIM_ts.T, FIM_ps.T, np.atleast_2d(FIM_ss)],
    ]) * N

    CRB_full = np.linalg.inv(FIM)
    CRB_full = 0.5 * (CRB_full + CRB_full.T)
    return CRB_full[:k, :k]
