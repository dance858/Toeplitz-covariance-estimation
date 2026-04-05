import ctypes
import platform
import pathlib
import numpy as np

_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    pkg_dir = pathlib.Path(__file__).parent
    system = platform.system()
    if system == "Darwin":
        name = "libnml.dylib"
    elif system == "Windows":
        name = "nml.dll"
    else:
        name = "libnml.so"

    lib_path = pkg_dir / name
    if not lib_path.exists():
        raise OSError(
            f"Could not find {name} in {pkg_dir}. "
            "Please install with: pip install nml-toeplitz"
        )

    _LIB = ctypes.CDLL(str(lib_path))

    _LIB.nml_new_solver.restype = ctypes.c_void_p
    _LIB.nml_new_solver.argtypes = [
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int,
    ]

    _LIB.nml_new_result.restype = ctypes.c_void_p
    _LIB.nml_new_result.argtypes = [ctypes.c_int]

    _LIB.nml_solve.restype = ctypes.c_int
    _LIB.nml_solve.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
    ]

    _LIB.nml_free_solver.restype = None
    _LIB.nml_free_solver.argtypes = [ctypes.c_void_p]

    _LIB.nml_free_result.restype = None
    _LIB.nml_free_result.argtypes = [ctypes.c_void_p]

    return _LIB


class _NML_result(ctypes.Structure):
    """Mirrors the C struct NML_result."""
    _fields_ = [
        ("x", ctypes.POINTER(ctypes.c_double)),
        ("y", ctypes.POINTER(ctypes.c_double)),
        ("grad_norm", ctypes.c_double),
        ("obj", ctypes.c_double),
        ("solve_time", ctypes.c_double),
        ("iter", ctypes.c_int),
        ("diag_init_succeeded", ctypes.c_int),
        ("num_of_hess_chol_fails", ctypes.c_int),
    ]


class NMLSolver:
    """Solver for ML Toeplitz covariance estimation.

    Minimizes  log det R + Tr(R^{-1} S)  subject to R being Toeplitz,
    where S is the sample covariance of the columns of Z.

    Parameters
    ----------
    n : int
        The covariance matrix has dimension n+1.
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
        self._lib = _load_lib()
        self._n = n
        self._ptr = self._lib.nml_new_solver(n, tol, beta, alpha, max_iter)

    def solve(self, Z, verbose=False):
        """Solve given a data matrix Z.

        Parameters
        ----------
        Z : np.ndarray, complex128, shape (n+1, K)
            Data matrix with K measurement vectors of dimension n+1.
        verbose : bool
            Print iteration progress.

        Returns
        -------
        dict with keys:
            'x' : np.ndarray, shape (n+1,) — real part of first column of R
            'y' : np.ndarray, shape (n,)   — imaginary part of first column of R
            'obj' : float                  — final objective value
            'grad_norm' : float            — gradient norm at solution
            'time' : float                 — solve time in seconds
            'iter' : int                   — number of Newton iterations
            'diag_init_succeeded' : bool
            'num_hess_chol_fails' : int
        """
        if self._ptr is None:
            raise RuntimeError("Solver has been freed")

        Z = np.asarray(Z, dtype=np.complex128, order="F")
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array of shape (n+1, K)")

        n = self._n
        n_plus_one, K = Z.shape
        if n_plus_one != n + 1:
            raise ValueError(
                f"Z has {n_plus_one} rows but solver expects {n + 1}"
            )

        Z_data = np.asfortranarray(Z).copy()

        result = _NML_result()
        x_buf = (ctypes.c_double * (n + 1))()
        y_buf = (ctypes.c_double * n)()
        result.x = ctypes.cast(x_buf, ctypes.POINTER(ctypes.c_double))
        result.y = ctypes.cast(y_buf, ctypes.POINTER(ctypes.c_double))

        ret = self._lib.nml_solve(
            self._ptr,
            Z_data.ctypes.data_as(ctypes.c_void_p),
            K, ctypes.byref(result), int(verbose),
        )

        if ret != 0:
            raise RuntimeError(f"NML solver returned error code {ret}")

        x = np.ctypeslib.as_array(result.x, shape=(n + 1,)).copy()
        y = np.ctypeslib.as_array(result.y, shape=(n,)).copy()

        return {
            "x": x,
            "y": y,
            "obj": result.obj,
            "grad_norm": result.grad_norm,
            "time": result.solve_time,
            "iter": result.iter,
            "diag_init_succeeded": bool(result.diag_init_succeeded),
            "num_hess_chol_fails": result.num_of_hess_chol_fails,
        }

    def free(self):
        """Free the underlying C solver."""
        if self._ptr is not None:
            self._lib.nml_free_solver(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    def __del__(self):
        self.free()
