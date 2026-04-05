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

    # int NML(double complex *Z_data, int n, int K, NML_out *output,
    #         double tol, double beta, double alpha, int verbose, int max_iter)
    _LIB.NML.restype = ctypes.c_int
    _LIB.NML.argtypes = [
        ctypes.c_void_p,   # Z_data (double complex *)
        ctypes.c_int,      # n
        ctypes.c_int,      # K
        ctypes.c_void_p,   # output (NML_out *)
        ctypes.c_double,   # tol
        ctypes.c_double,   # beta
        ctypes.c_double,   # alpha
        ctypes.c_int,      # verbose
        ctypes.c_int,      # max_iter
    ]

    _LIB.NML_free_output.restype = None
    _LIB.NML_free_output.argtypes = [ctypes.c_void_p]

    return _LIB


class _NML_out(ctypes.Structure):
    """Mirrors the C struct NML_out."""
    _fields_ = [
        ("grad_norm", ctypes.c_double),
        ("obj", ctypes.c_double),
        ("iter", ctypes.c_int),
        ("diag_init_succeded", ctypes.c_int),
        ("num_of_hess_chol_fails", ctypes.c_int),
        ("total_time", ctypes.c_double),
        ("x_sol", ctypes.POINTER(ctypes.c_double)),
        ("y_sol", ctypes.POINTER(ctypes.c_double)),
    ]


def solve(Z, tol=1e-8, beta=0.8, alpha=0.05, verbose=False, max_iter=200):
    """Solve the ML Toeplitz covariance estimation problem.

    Minimizes  log det R + Tr(R^{-1} S)  subject to R being Toeplitz,
    where S is the sample covariance of the columns of Z.

    Parameters
    ----------
    Z : np.ndarray, complex128, shape (n+1, K)
        Data matrix with K measurement vectors of dimension n+1.
    tol : float
        Convergence tolerance on Newton decrement.
    beta : float
        Backtracking line search shrinkage factor.
    alpha : float
        Backtracking line search sufficient decrease parameter.
    verbose : bool
        Print iteration progress.
    max_iter : int
        Maximum Newton iterations.

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
    lib = _load_lib()

    Z = np.asarray(Z, dtype=np.complex128, order="F")
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array of shape (n+1, K)")

    n_plus_one, K = Z.shape
    n = n_plus_one - 1

    # Make a contiguous copy (column-major) since C may modify it
    Z_data = np.asfortranarray(Z).copy()

    out = _NML_out()
    ret = lib.NML(
        Z_data.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n),
        ctypes.c_int(K),
        ctypes.byref(out),
        ctypes.c_double(tol),
        ctypes.c_double(beta),
        ctypes.c_double(alpha),
        ctypes.c_int(int(verbose)),
        ctypes.c_int(max_iter),
    )

    if ret != 0:
        raise RuntimeError(f"NML solver returned error code {ret}")

    # Copy results before freeing
    x = np.ctypeslib.as_array(out.x_sol, shape=(n + 1,)).copy()
    y = np.ctypeslib.as_array(out.y_sol, shape=(n,)).copy()

    result = {
        "x": x,
        "y": y,
        "obj": out.obj,
        "grad_norm": out.grad_norm,
        "time": out.total_time,
        "iter": out.iter,
        "diag_init_succeeded": bool(out.diag_init_succeded),
        "num_hess_chol_fails": out.num_of_hess_chol_fails,
    }

    lib.NML_free_output(ctypes.byref(out))

    return result
