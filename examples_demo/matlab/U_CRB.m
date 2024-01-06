function [CRB] = U_CRB(P, theta_rad, sig2, m, d, N)

%    P      <- The covariance matrix of the source signals
%    N      <- number of snapshots to generate
%    sig2   <- noise variance
%    m      <- number of sensors
%    d      <- sensor spacing in wavelengths 


j=sqrt(-1);
A=exp(-2*pi*j*d*[0:m-1].'*sin(theta_rad));
D = (-2*pi*j*d*[0:m-1].'*cos(theta_rad)).*exp(-2*pi*j*d*[0:m-1].'*sin(theta_rad));

% Covariance matrix of array output.
R = A*P*A' + sig2*eye(m);

proj_A_orth = eye(m) - A*((A'*A)\A');

CRB = sig2/(2*N)*inv(real((D'*proj_A_orth*D).*transpose((P*A'*(R\A)*P))));

end