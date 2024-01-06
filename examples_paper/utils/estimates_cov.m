function [sample_cov, DA_out, AML_out, NML_out] =  estimates_cov(Y)

[n, K] = size(Y);
n = n - 1;
% sample covariance
sample_cov = 1/K*(Y*Y');

verbose = false;
max_iter = 200;
tol = 1e-8;
beta = 0.8;
alpha = 0.05;

% apply NML.
[x, y, grad_norm, ~, NML_time, iter] = ...
            NML(real(Y(:)), imag(Y(:)), n, K, verbose, tol, beta, alpha, max_iter);
if iter >= 100
    fprintf("Number of Newton iterations: %i \n", iter)
    fprintf("Number of Newton iterations: %i \n", grad_norm)
end
NML_out.estimate = toeplitz([2*x(1); x(2:end) + 1i*y]);
NML_out.solve_time = NML_time;
NML_out.iter = iter;

% apply diagonal averaging
tic;
[DA_cov, ~, ~] = AAD(sample_cov);
DA_out.solve_time = toc;
DA_out.estimate = DA_cov;

% apply AML
if K >= n+1
    tic;
    [Psi_AML] = create_Psi_AML(size(sample_cov, 1));
    [AML_cov] = AML(sample_cov, K, Psi_AML);
    AML_out.solve_time = toc;
    AML_out.estimate = AML_cov.estimate;
else
    AML_out = -1;
end

end