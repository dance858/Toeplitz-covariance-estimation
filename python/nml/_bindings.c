#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "nml/NML_solver.h"

#define NML_SOLVER_CAPSULE "nml.solver"

/* ---- Capsule destructor ---- */
static void solver_capsule_destructor(PyObject *capsule)
{
    NML_solver *solver =
        (NML_solver *) PyCapsule_GetPointer(capsule, NML_SOLVER_CAPSULE);
    if (solver)
    {
        nml_free_solver(solver);
    }
}

/* ---- new_solver(n, tol, beta, alpha, max_iter) -> capsule ---- */
static PyObject *py_new_solver(PyObject *self, PyObject *args)
{
    int n, max_iter;
    double tol, beta, alpha;

    if (!PyArg_ParseTuple(args, "idddi", &n, &tol, &beta, &alpha, &max_iter))
    {
        return NULL;
    }

    NML_solver *solver = nml_new_solver(n, tol, beta, alpha, max_iter);
    if (!solver)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NML solver");
        return NULL;
    }

    return PyCapsule_New(solver, NML_SOLVER_CAPSULE, solver_capsule_destructor);
}

/* ---- solve(solver_capsule, Z, verbose) -> dict ---- */
static PyObject *py_solve(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    PyArrayObject *Z_array;
    int verbose;

    if (!PyArg_ParseTuple(args, "OO!i", &capsule, &PyArray_Type, &Z_array,
                          &verbose))
    {
        return NULL;
    }

    NML_solver *solver =
        (NML_solver *) PyCapsule_GetPointer(capsule, NML_SOLVER_CAPSULE);
    if (!solver)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid or freed solver handle");
        return NULL;
    }

    /* Ensure Z is complex128, Fortran-contiguous */
    PyArrayObject *Z_f = (PyArrayObject *) PyArray_FromAny(
        (PyObject *) Z_array, PyArray_DescrFromType(NPY_COMPLEX128), 2, 2,
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!Z_f)
    {
        return NULL;
    }

    int n_plus_one = (int) PyArray_DIM(Z_f, 0);
    int K = (int) PyArray_DIM(Z_f, 1);
    int n = n_plus_one - 1;

    nml_complex *Z_data = (nml_complex *) PyArray_DATA(Z_f);

    NML_result *result = nml_new_result(n);
    if (!result)
    {
        Py_DECREF(Z_f);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate NML result");
        return NULL;
    }

    int ret = nml_solve(solver, Z_data, K, result, verbose);
    Py_DECREF(Z_f);

    if (ret != 0)
    {
        nml_free_result(result);
        PyErr_Format(PyExc_RuntimeError, "NML solver returned error code %d",
                     ret);
        return NULL;
    }

    /* Build x and y as NumPy arrays */
    npy_intp x_dim = n + 1;
    npy_intp y_dim = n;
    PyObject *x_arr = PyArray_SimpleNew(1, &x_dim, NPY_DOUBLE);
    PyObject *y_arr = PyArray_SimpleNew(1, &y_dim, NPY_DOUBLE);
    if (!x_arr || !y_arr)
    {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        nml_free_result(result);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) x_arr), result->x,
           sizeof(double) * (n + 1));
    memcpy(PyArray_DATA((PyArrayObject *) y_arr), result->y,
           sizeof(double) * n);

    PyObject *dict = Py_BuildValue(
        "{s:N, s:N, s:d, s:d, s:d, s:i, s:O, s:i}", "x", x_arr, "y", y_arr,
        "obj", result->obj, "grad_norm", result->grad_norm, "time",
        result->solve_time, "iter", result->iter, "diag_init_succeeded",
        result->diag_init_succeeded ? Py_True : Py_False, "num_hess_chol_fails",
        result->num_of_hess_chol_fails);

    nml_free_result(result);
    return dict;
}

/* ---- free_solver(solver_capsule) ---- */
static PyObject *py_free_solver(PyObject *self, PyObject *args)
{
    PyObject *capsule;

    if (!PyArg_ParseTuple(args, "O", &capsule))
    {
        return NULL;
    }

    NML_solver *solver =
        (NML_solver *) PyCapsule_GetPointer(capsule, NML_SOLVER_CAPSULE);
    if (solver)
    {
        nml_free_solver(solver);
        /* Remove destructor so it won't double-free on GC */
        PyCapsule_SetDestructor(capsule, NULL);
    }

    Py_RETURN_NONE;
}

/* ---- Module definition ---- */
static PyMethodDef nml_methods[] = {
    {"new_solver", py_new_solver, METH_VARARGS, "Create an NML solver"},
    {"solve", py_solve, METH_VARARGS, "Solve the Toeplitz ML problem"},
    {"free_solver", py_free_solver, METH_VARARGS, "Free an NML solver"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef nml_module = {
    PyModuleDef_HEAD_INIT, "_nml", NULL, -1, nml_methods};

PyMODINIT_FUNC PyInit__nml(void)
{
    import_array();
    return PyModule_Create(&nml_module);
}
