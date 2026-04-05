function [Y, true_cov] = generate_ula_data(power_source, sig2, d, m, K, N, ...
                                           wavelength, theta_rad)    

s = sqrt(power_source/2)*(randn(K, N)+1j*randn(K, N));
e = sqrt(sig2/2)*(randn(m, N)+1j*randn(m,N));
A = exp(-2j*pi/wavelength*(d*[0:m-1].'*sin(theta_rad)));
Y = A*s+e;

true_cov = power_source*(A*A') + sig2*eye(m);
end