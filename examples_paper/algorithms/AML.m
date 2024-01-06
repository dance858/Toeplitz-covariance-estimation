function [out] = AML(R_tilde, N, Psi)
%
% INPUT: 
%         R_tilde: sample covariance of dimension M.
%               N: Number of data points used to build the sample covariance.
% OUTPUT:
%         out.x: real part first row of Toeplitz matrix
%         out.y: imaginary part first row of Toeplitz matrix
%         out.T: Toeplitz matrix
%
% Note:
% *  If out.x = (x0, x1, x2, ..., xm), out.y = (y1, y2, ..., ym), then
%    out.T is the Hermitian Toeplitz matrix with first row equal to
%    (x0, x1+1i*y1, x2+1i*y2, ..., xm+1i*ym).
% out.T = [x0   x1+1i*y1 
% * This estimator assumes that N >= m.

tic;
M = size(R_tilde, 1);
r_tilde = R_tilde(:);
C_tilde =  1/N*kron(transpose(R_tilde), R_tilde);
C_tilde_chol = chol(C_tilde, 'lower');
%phi_tilde = (Psi'*(C_tilde\Psi))\(Psi'*(C_tilde\r_tilde));
phi_tilde = (Psi'*(C_tilde_chol'\(C_tilde_chol\Psi)))\(Psi'*(C_tilde_chol'\(C_tilde_chol\r_tilde)));
phi_tilde = real(phi_tilde); % Remove negligible imaginary parts.
x = [phi_tilde(1); phi_tilde(2:2:2*M-2)];
y = phi_tilde(3:2:2*M-1);
T = toeplitz(x + 1i*[0;y]);
out.x = x;
out.y = y;
out.estimate = T;
out.solve_time = toc;
end