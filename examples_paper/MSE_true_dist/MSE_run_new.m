clear; clc;
setenv('BLAS_VERSION', '/usr/lib/x86_64-linux-gnu/libblas.so');
addpath('../algorithms/NML')
addpath('../algorithms/other_alg')
addpath('../utils')
%% 
n = 15; samples = (20:20:200);
MC_runs = 5;

% Generate true covariance matrix
rng(0)
[true_cov] = toeplitz_via_cross_corr(n);



%%
[MSE_NML, MSE_NML_mex, MSE_AML, MSE_DA, MSE_ATOM, ...
 NML_mean_time, NML_mex_mean_time, AML_mean_time, ...
 DA_mean_time, ATOM_mean_time] = MSE_compute_new(samples, true_cov, MC_runs);

%%
% Compute CRB
CRB = zeros(1, length(samples));
for k = 1:length(samples)
   N = samples(k); 
   FIM = CRB_true_dist(true_cov, N, n);
   CRB(k) = real(trace(inv(FIM)));
end


