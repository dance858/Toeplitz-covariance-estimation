function [X] = generate_samples(T, N)
% Generates N samples of a complex-valued random vector with
% a circular Gaussian distribution with zero mean and covariance matrix
% T.
% INPUT:
%        T: true covariance matrix (Toeplitz)
%        N: number of samples

n = size(T, 1) - 1;
chol_T = chol(T, 'lower');
X = chol_T*(randn(n+1, N) +1i*randn(n+1, N))/sqrt(2);
end