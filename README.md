# Maximum likelihood Toeplitz covariance estimation
This repository hosts an implementation of Newton's method for solving the maximum likelihood estimation problem for a covariance matrix that is known to be Toeplitz:

$$
\begin{array}{r}
\text{minimize} & \textbf{log det } R + \textbf{Tr}(R^{-1} S) \\
\text{subject to} & R \text{ being Toeplitz} \hspace{1.2cm}
\end{array}
$$

with decision variable
$R \in \mathbf{H}^{n+1}$
and problem data
$S \in \mathbf{H}^{n+1}$ representing the sample covariance matrix. Here
$\mathbf{H}^{n+1}$ is the space of Hermitian matrices of dimension $n + 1$. A
description of the method, which we refer to as `NML`,  along with possible
applications can be found in our [paper](https://www.sciencedirect.com/science/article/pii/S0165168424001257).

## Installation

### Python <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="20" />
**Prerequisites:** [FFTW3](https://www.fftw.org/) must be installed on your system.
- macOS: `brew install fftw`
- Linux: `sudo apt install libfftw3-dev`

Then install with pip:
```
pip install .
```

### MATLAB <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matlab/matlab-original.svg" height="20"/>
Build the MEX interface with CMake:
```
cmake -B build -DBUILD_MATLAB=ON
cmake --build build
```
The compiled MEX file will be in `build/matlab/`.

### Building the C library from source <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/c/c-original.svg" height="20"/>
`NML` is written in C and uses [FFTW3](https://www.fftw.org/) for fast Fourier transforms. On macOS it uses the Accelerate framework for BLAS/LAPACK; on Linux it requires [CBLAS](https://www.netlib.org/blas/#_cblas) and [LAPACKE](https://www.netlib.org/lapack/lapacke.html).
```
cmake -B build
cmake --build build
```

## Examples

### Python <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="20" />
```python
import numpy as np
from nml import solve

# Z is a (n+1, K) complex data matrix
result = solve(Z)
x, y = result["x"], result["y"]

# Reconstruct the Toeplitz covariance matrix
from scipy.linalg import toeplitz
R_hat = toeplitz(np.concatenate([[2*x[0]], x[1:] - 1j*y]))
```

See `examples/python/demo.py` for a full example.

### MATLAB <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matlab/matlab-original.svg" height="20"/>
The MATLAB interface exposes the function
```
[x, y, grad_norm, obj, solve_time, iter] = NML(real_Z, imag_Z, n, K, verbose, tol, beta, alpha, max_iter)
```
The input parameters are defined as follows.
* `real_Z`- the real part of the data points, stacked along columns
* `imag_Z`- the imaginary part of the data points, stacked along columns
* `n` - the covariance matrix to be estimated has dimension $n + 1$
* `K` - the number of measurements
* `verbose` - if true the progress is printed out in every iteration
* `tol` - the algorithm terminates when the Newton decrement is smaller than `tol`
* `beta` and `alpha`  - backtracking parameters,
* `max_iter` - maximum number of iterations

Typical values are `tol` = $10^{-8}$, `beta` = $0.8$ and `alpha` = $0.05.$

The output parameters are defined as follows.
* `x` and `y` - The real and imaginary parts of the maximum likelihood Toeplitz estimate. The corresponding covariance matrix can be reconstructed with       the command `toeplitz([2*x(1); x(2:end) + 1i*y])`
* `grad_norm` - Euclidean norm of the gradient
* `obj` - objective value
* `solve_time` - solve time in seconds
* `iter` - number of iterations

An example of how this function is called is given in `examples/matlab/demo.m`. Running `demo.m` results in the following figure:

$\hspace{3.5cm}$ ![](https://github.com/dance858/Toeplitz-covariance-estimation/blob/main/demo.jpg)

This figure shows the mean-squared estimation error for MUSIC when used with the sample covariance matrix (labelled with `MSE_SC`) and the maximum likelihood estimate (labelled with `MSE_NML`), versus the number of measurements $K$. For more explanations and details on the dotted lines (which represent Cramér-Rao bounds) we refer to Section 4 of our paper.

## Citing
If you find this repository useful, please consider giving it a star. If you
wish to cite this work you may use the following BibTex:

```
@article{Ced24,
title = {Toeplitz covariance estimation with applications to MUSIC},
journal = {Signal Processing},
volume = {221},
pages = {109506},
year = {2024},
issn = {0165-1684},
doi = {https://doi.org/10.1016/j.sigpro.2024.109506},
author = {Daniel Cederberg},
}
```
