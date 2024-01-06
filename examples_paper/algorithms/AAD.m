function [estimate, x, y] = AAD(S)
% Averages diagonals. Note that x(1) is NOT scaled by 0.5.
n = size(S, 1) - 1;
z = zeros(n+1, 1);
for k = 0:n
   z(k+1) = mean(diag(S, k)); 
end
x = real(z);
y = imag(z(2:n+1));
estimate = toeplitz([x(1); x(2:end)+1i*y]);
end