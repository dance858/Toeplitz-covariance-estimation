function [T] = toeplitz_via_cross_corr(n)
% Generates a positive definite complex-valued Toeplitz matrix of size
% (n+1)x(n+1).
x = randn(1, n+1) + 1i*randn(1, n+1);
M = 2*(n+1) - 1;
z_padded = ifft(fft(x, M).*conj(fft(x, M)))/M;
z_via_fft = z_padded(1:n+1);
z_via_fft(1) = real(z_via_fft(1));
T = toeplitz(z_via_fft);

end