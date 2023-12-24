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
$S \in \mathbf{H}^{n+1}.$ Here $\mathbf{H}^{n+1}$ is the space of Hermitian matrices of dimension $n + 1$. A description of the method, which we refer to as `NML`, and possible applications can be found in the paper XXX.

## Installation

### MATLAB <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matlab/matlab-original.svg" height="20"/> 

### Python <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="20" />

### Building from source <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/c/c-original.svg" height="20"/>
`NML` is written in C and uses [`BLAS`](https://www.netlib.org/blas/) and [`LAPACK`](https://www.netlib.org/lapack/lapacke.html) for linear algebra operations, 
and [`FFTW3`](https://www.fftw.org/) to compute fast Fourier transforms. After having installed these packages you can build `NML` using the provided CMake configuration.

## Examples
In this example we show how the maximum-likelihood Toeplitz covariance estimate can be used to enhance the performance of the multiple-signal classification algorithm (MUSIC) for direction-of-arrival estimation.

### MATLAB <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matlab/matlab-original.svg" height="20"/> 
The MATLAB interface exposes the function
```
[output] = NML(input)
```

An example of how it is called is given in `demo.m` in the `examples`-folder. Running `demo.m` results in the following figure:

![]((https://github.com/dance858/Toeplitz-covariance-estimation/blob/main/demo.jpg?raw=true))

![](https://github.com/cvxgrp/pymde/blob/main/images/mnist_std.png?raw=true)

### Python <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="20" /> 
The Python interface exposes the function
```
[output] = NML(input)
```
An example of how it is called is given in `demo.py`.

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
