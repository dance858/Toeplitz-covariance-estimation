function [out] = ATOM(S, gamma0, X)
% Solves the maximum likelihood estimation problem using ATOM.
%
% INPUT:
%         S: sample covariance
% OUTPUT: 
%       out.solve_time: execution time ATOM.
%      
%     

% Initialization by averaging diagonals of sample covariance. The
% corresponding Toeplitz matrix toeplitz([2*x(1); x(2:end) + 1i*y])
% is not guaranteed to be positive definite.  
[x, y] = initialization(S, X, true);
n = length(y);

max_iter = 50; K = 5;

tic;
[estimate, obj, ~] = ATOM2(n+1, S, toeplitz([2*x(1); x(2:n+1) + 1i*y]), ...
                         max_iter, K, gamma0);
out.solve_time = toc;
out.estimate = estimate;
out.ML_obj =  obj(end);
end