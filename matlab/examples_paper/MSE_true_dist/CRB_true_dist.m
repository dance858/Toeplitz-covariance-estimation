function [FIM] = CRB_own(true_cov, N, n)

FIM = zeros(2*n+1, 2*n+1);

for i = 1:2*n+1
    if i == 1
        T_partial_theta_i = eye(n+1);
    elseif i >= 2 && i <= n+1
        E_i_minus_one = diag(ones(n+1-(i-1), 1),-(i-1));
        T_partial_theta_i = E_i_minus_one' + E_i_minus_one;
    elseif i >= n+2 
        E_i_minus_n_minus_one = diag(ones(n+1-(i-n-1), 1),-(i-n-1));
        T_partial_theta_i = 1i*(E_i_minus_n_minus_one' - E_i_minus_n_minus_one);
    end
    for j=1:2*n+1
        if j == 1
            T_partial_theta_j = eye(n+1);
        elseif j >= 2 && j <= n+1
            E_j_minus_one = diag(ones(n+1-(j-1), 1),-(j-1));
            T_partial_theta_j = E_j_minus_one' + E_j_minus_one;
        elseif j >= n+2 
            E_j_minus_n_minus_one = diag(ones(n+1-(j-n-1), 1),-(j-n-1));
            T_partial_theta_j = 1i*(E_j_minus_n_minus_one' - E_j_minus_n_minus_one);
        end
        FIM(i, j) = N*trace((true_cov\T_partial_theta_i)*(true_cov\T_partial_theta_j));
    end
end



end