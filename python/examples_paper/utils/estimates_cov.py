"""Wrapper that estimates a Toeplitz covariance using multiple methods."""
import time
import numpy as np
from scipy.linalg import toeplitz
from nml import NMLSolver

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algorithms.aad import aad
from algorithms.aml import aml, create_psi_aml


def estimates_cov(Y):
    """Estimate covariance from data using SC, DA, AML, and NML.

    Parameters
    ----------
    Y : ndarray, shape (n, K)
        Complex data matrix.

    Returns
    -------
    sample_cov : ndarray, shape (n, n)
    da_out : dict with 'estimate', 'x', 'y'
    aml_out : dict (or None if K < n)
    nml_out : dict with 'estimate', 'x', 'y', 'solve_time', 'iter'
    """
    n, K = Y.shape

    # Sample covariance
    sample_cov = (Y @ Y.conj().T) / K

    # NML
    solver = NMLSolver(n)
    result = solver.solve(Y, verbose=False)
    solver.free()
    x, y = result["x"], result["y"]
    nml_cov = toeplitz(np.concatenate([[2 * x[0]], x[1:] - 1j * y]))
    nml_out = {
        "estimate": nml_cov,
        "x": x,
        "y": y,
        "solve_time": result["time"],
        "iter": result["iter"],
    }

    # DA (Averaging Along Diagonals)
    t0 = time.perf_counter()
    da_out = aad(sample_cov)
    da_out["solve_time"] = time.perf_counter() - t0

    # AML (requires K >= n)
    if K >= n:
        Psi = create_psi_aml(n)
        aml_out = aml(sample_cov, K, Psi)
    else:
        aml_out = None

    return sample_cov, da_out, aml_out, nml_out
