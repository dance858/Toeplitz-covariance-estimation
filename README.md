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
$S \in \mathbf{H}^{n+1}.$ Here $\mathbf{H}^{n+1}$ is the space of Hermitian matrices of dimension $n + 1$. A description of the method, which we refer to as `NML`,  along with possible applications can be found in our paper XXX.

## Installation

### MATLAB <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matlab/matlab-original.svg" height="20"/> 
The MATLAB installation assumes that [`CBLAS`](https://www.netlib.org/blas/#_cblas), [`LAPACKE`](https://www.netlib.org/lapack/lapacke.html) and [`FFTW3`](https://www.fftw.org/) are installed on your system. After having installed these dependencies `NML` can be built with
```
git clone https://github.com/dance858/Toeplitz-covariance-estimation.git
make -f make_matlab
```
This will create a mex-file. Remember to include the path to the mex-file where you want to use it.

If you don't have the dependencies above installed XXX
### Python <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="20" />
```
pip install NML
```

### Building from source <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/c/c-original.svg" height="20"/>
`NML` is written in C and uses [`CBLAS`](https://www.netlib.org/blas/#_cblas) and [`LAPACKE`](https://www.netlib.org/lapack/lapacke.html) for linear algebra operations, 
and [`FFTW3`](https://www.fftw.org/) to compute fast Fourier transforms. After having installed these packages you can build `NML` using the provided CMake configuration.

## Examples
In this example we show how the maximum-likelihood Toeplitz covariance estimate can be used to enhance the performance of the multiple-signal classification algorithm (MUSIC) for direction-of-arrival estimation.

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
* `real_Z`- the real part of the data points, stacked along columns
* `imag_Z`- the imaginary part of the data points, stacked along columns
  
An example of how this function is called is given in `demo.m` in the `examples`-folder. Running `demo.m` results in the following figure:

$\hspace{3.5cm}$ ![](https://github.com/dance858/Toeplitz-covariance-estimation/blob/main/demo.jpg)

This figure shows the mean-squared estimation error for MUSIC when used with the sample covariance matrix (labelled with `MSE_SC`) and the maximum likelihood estimate (labelled with `MSE_NML`), versus the number of measurements $K$. For more explanations and details on the dotted lines (which represent Cramér-Rao bounds) we refer to Section 4 of our paper. 

### Python <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="20" /> 
The Python interface exposes the function
```
[x, y, grad_norm, obj, solve_time, iter] = NML(real_Z, imag_Z, n, K, verbose, tol, beta, alpha, max_iter)
```
The parameters are defined as in the MATLAB interface. An example of how this function is called is given in `demo.py`.

## Citing
If you find this repository useful, please consider giving it a star.

If you wish to cite our work we ask you to use the following BibTex entry.

```
@article{CederbergXXX,
  author  = {Daniel Cederberg},
  title   = {XXX},
  journal = {XXX},
  year    = {XXX},
}
```
