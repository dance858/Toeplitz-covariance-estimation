import numpy as np
from nml import _nml


class NMLSolver:
    """Solver for ML Toeplitz covariance estimation.

    Minimizes  log det R + Tr(R^{-1} S)  subject to R being Toeplitz,
    where S is the sample covariance of the columns of Z.

    Parameters
    ----------
    n : int
        Dimension of the covariance matrix.
    tol : float
        Convergence tolerance on Newton decrement.
    beta : float
        Backtracking line search shrinkage factor.
    alpha : float
        Backtracking line search sufficient decrease parameter.
    max_iter : int
        Maximum Newton iterations.
    """

    def __init__(self, n, tol=1e-8, beta=0.8, alpha=0.05, max_iter=200):
        self._n = n
        self._solver = _nml.new_solver(n - 1, tol, beta, alpha, max_iter)

    def solve(self, Z, verbose=False):
        """Solve given a data matrix Z.

        Parameters
        ----------
        Z : np.ndarray, complex128, shape (n, K)
            Data matrix with K measurement vectors of dimension n.
        verbose : bool
            Print iteration progress.

        Returns
        -------
        dict with keys:
            'x' : np.ndarray, shape (n,)   — real part of first column of R
            'y' : np.ndarray, shape (n-1,) — imaginary part of first column of R
            'obj' : float                  — final objective value
            'grad_norm' : float            — gradient norm at solution
            'time' : float                 — solve time in seconds
            'iter' : int                   — number of Newton iterations
            'diag_init_succeeded' : bool
            'num_hess_chol_fails' : int
        """
        if self._solver is None:
            raise RuntimeError("Solver has been freed")

        Z = np.asarray(Z, dtype=np.complex128, order="F")
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array of shape (n, K)")
        if Z.shape[0] != self._n:
            raise ValueError(
                f"Z has {Z.shape[0]} rows but solver expects {self._n}"
            )

        return _nml.solve(self._solver, Z, int(verbose))

    def free(self):
        """Free the underlying C solver."""
        if self._solver is not None:
            _nml.free_solver(self._solver)
            self._solver = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    def __del__(self):
        self.free()
