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
The compiled MEX file will be in `build/matlab/`. Add this directory to your MATLAB path.

### Building the C library from source <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/c/c-original.svg" height="20"/>
`NML` is written in C and uses [FFTW3](https://www.fftw.org/) for fast Fourier transforms. On macOS it uses the Accelerate framework for BLAS/LAPACK; on Linux it requires [CBLAS](https://www.netlib.org/blas/#_cblas) and [LAPACKE](https://www.netlib.org/lapack/lapacke.html).
```
cmake -B build
cmake --build build
```

## Usage

The solver follows a create-solve-free pattern. First create a solver for a given problem size and algorithmic parameters, then call solve (potentially multiple times with different data), and finally free the solver.

### Python <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="20" />
```python
import numpy as np
from nml import NMLSolver

# Create solver for dimension n+1
solver = NMLSolver(n)

# Z is a (n+1, K) complex data matrix
result = solver.solve(Z)
x, y = result["x"], result["y"]

# Reconstruct the Toeplitz covariance matrix
from scipy.linalg import toeplitz
R_hat = toeplitz(np.concatenate([[2*x[0]], x[1:] - 1j*y]))

solver.free()
```

### MATLAB <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matlab/matlab-original.svg" height="20"/>
```matlab
% Create solver for dimension n+1
solver = nml_new_solver(n, tol, beta, alpha, max_iter);

% Solve: Z is a (n+1) x K complex data matrix
[x, y, grad_norm, obj, solve_time, iter] = nml_solve(solver, Z, verbose);
R_hat = toeplitz([2*x(1); x(2:end) + 1i*y]);

% Free solver
nml_free_solver(solver);
```

## Example

Running `python/examples/demo.py` produces the following figure, which compares the mean-squared estimation error for MUSIC when using the sample covariance matrix (`MSE_SC`) versus the NML Toeplitz estimate (`MSE_NML`), as a function of the number of measurements $K$. The dotted lines show the unconditional and conditional Cramér-Rao bounds. For details we refer to Section 4 of our [paper](https://www.sciencedirect.com/science/article/pii/S0165168424001257).

$\hspace{3.5cm}$ ![](https://github.com/dance858/Toeplitz-covariance-estimation/blob/main/python/examples/demo.png)

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
