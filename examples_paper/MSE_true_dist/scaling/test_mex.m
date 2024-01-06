clear; clc;
addpath('../../algorithms/NML')
addpath('../../algorithms/other_alg')
addpath('../../utils')
%%

n = 15;
K = n+1;
[true_cov] = toeplitz_via_cross_corr(n);

X = generate_samples(true_cov, K);
S = 1/K*(X*X');
%%
tic;
[out_NML] = NML(S, X, 2);
toc;

%%

% Extract lower triangular part.
Zvec = X(:);

tic;
[x_sol, y_sol, grad_norm, obj, tot_time_NML, iter, diag_init_success, ...
 num_of_hess_chol_fails] ...
    = NML_mex(real(Zvec), imag(Zvec), n, K);
toc;

