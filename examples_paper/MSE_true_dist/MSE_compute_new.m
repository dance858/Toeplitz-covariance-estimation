function [MSE_NML, MSE_NML_mex, MSE_AML, MSE_DA, MSE_ATOM, ...
         NML_mean_time, NML_mex_mean_time, AML_mean_time, ...
         DA_mean_time, ATOM_mean_time] = MSE_compute_new(samples, true_cov, MC_runs)


MSE_NML = zeros(1, length(samples));
MSE_NML_mex = zeros(1, length(samples));
MSE_AML = zeros(1, length(samples));
MSE_DA = zeros(1, length(samples));
MSE_ATOM = zeros(1, length(samples));


NML_mean_time = 0;
NML_mex_mean_time = 0;
AML_mean_time = 0;
DA_mean_time = 0;
ATOM_mean_time = 0;

n = size(true_cov, 1) - 1;
first_row_real = real(true_cov(1, :));
first_row_imag = imag(true_cov(1, 2:end));
gamma0 = 0.1;


for ii = 1:length(samples) 
    N = samples(ii);
    
    for run = 1:MC_runs
        X = generate_samples(true_cov, N);
        S = 1/N*(X*X');
        
        % Estimate covariance matrix.
        [~, DA_out, AML_out, NML_out, NML_mex_out] =  estimates_cov(X);
        [ATOM_out] = ATOM(S, gamma0, X);


        DA_cov = DA_out.estimate;
        NML_cov = NML_out.estimate;
        NML_mex_cov = NML_mex_out.estimate;
        ATOM_cov = ATOM_out.estimate;

        if norm(NML_cov - NML_mex_cov, 'fro') > 1e-6
            fprintf("Large difference. Pause.")
        end
        
        MSE_NML(ii) = MSE_NML(ii) + norm(first_row_real - real(NML_cov(1, :)))^2 ...
                                  + norm(first_row_imag - imag(NML_cov(1, 2:n+1)))^2;
        MSE_NML_mex(ii) = MSE_NML_mex(ii) + norm(first_row_real - real(NML_mex_cov(1, :)))^2 ...
                                  + norm(first_row_imag - imag(NML_mex_cov(1, 2:n+1)))^2;
        MSE_DA(ii) = MSE_DA(ii) + norm(first_row_real - real(DA_cov(1, :)))^2 ...
                                  + norm(first_row_imag - imag(DA_cov(1, 2:n+1)))^2;
        MSE_ATOM(ii) = MSE_ATOM(ii) + norm(first_row_real - real(ATOM_cov(1, :)))^2 ...
                                  + norm(first_row_imag - imag(ATOM_cov(1, 2:n+1)))^2;
        
        fprintf("Time (s) DA/NML/NML_mex/ATOM: %f, %f, %f, %f \n", ...
                DA_out.solve_time, NML_out.solve_time, NML_mex_out.solve_time, ...
                ATOM_out.solve_time)

        DA_mean_time = DA_mean_time + DA_out.solve_time;
        NML_mean_time = NML_mean_time + NML_out.solve_time;
        NML_mex_mean_time = NML_mex_mean_time + NML_mex_out.solve_time;
        ATOM_mean_time = ATOM_mean_time + ATOM_out.solve_time;
        
        if N >= n+1
            AML_mean_time = AML_mean_time + AML_out.solve_time;
            AML_cov = AML_out.estimate;
            MSE_AML(ii) = MSE_AML(ii) + norm(first_row_real - real(AML_cov(1, :)))^2 ...
                                  + norm(first_row_imag - imag(AML_cov(1, 2:n+1)))^2;
        end
    end
end


NML_mean_time = NML_mean_time/(length(samples)*MC_runs);
NML_mex_mean_time = NML_mex_mean_time/(length(samples)*MC_runs);
ATOM_mean_time = ATOM_mean_time/(length(samples)*MC_runs);
DA_mean_time = DA_mean_time/(length(samples)*MC_runs);
AML_mean_time = AML_mean_time/(sum(samples >= n+1)*MC_runs);

MSE_NML = MSE_NML/MC_runs;
MSE_NML_mex = MSE_NML_mex/MC_runs;
MSE_ATOM = MSE_ATOM/MC_runs;
MSE_DA = MSE_DA/MC_runs;
MSE_AML = MSE_AML/MC_runs;
end
