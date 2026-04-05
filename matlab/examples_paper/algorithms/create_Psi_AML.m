function [Psi] = create_Psi_AML(M)
    Omega = zeros(2*M-1, 2*M-1); Omega(1, 1) = 1; col_counter = 2;
    for row = 2:2*M-1
       % even row
       if rem(row, 2) == 0
           Omega(row, col_counter) = 1;
           Omega(row, col_counter+1) = 1i;
      % odd row
       else
           Omega(row, col_counter) = 1;
           Omega(row, col_counter+1) = -1i;
           col_counter = col_counter + 2;
       end
    end

    % Create matrix denoted by Sigma in Li98.
    Sigma = zeros(M^2, 2*M-1);
    IM = eye(M);
    Sigma(:, 1) = IM(:);
    col_counter = 2;
    for m = 1:M-1
        Q_col = [zeros(M-m, m), eye(M-m); zeros(m, M-m), zeros(m, m)];
        Q_col_T = Q_col';
        Sigma(:, col_counter) = Q_col(:);
        Sigma(:, col_counter+1) = Q_col_T(:);
        col_counter = col_counter + 2;
    end
Psi = Sigma*Omega; 
end