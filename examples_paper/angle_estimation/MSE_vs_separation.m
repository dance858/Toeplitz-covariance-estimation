% compares the performance of ULA with the CRB.
clear; clc;
setenv('BLAS_VERSION', '/usr/lib/x86_64-linux-gnu/libblas.so');
addpath('../algorithms/')
addpath('../utils')
%% Array geometry
M = 5;                                        % number of sources
snr = 10;                                      % signal-to-noise ratio
m = 20;                                       % number of sensors
power_source = 1;  
P = power_source*eye(M);                      % covariance matrix for source signals
sig2 = power_source*10^(-snr/10);             % noise variance
wavelength = 1;                               % normalized wavelength
d = wavelength/2;                             % spacing between sensors, in wavelength  

% Experiment parameters
d_theta = linspace(2, 8, 20);
MC_runs = 250;
K = 200;

% Containers for evaluating the performance
MSE_SC = zeros(1, length(d_theta));
MSE_NML = zeros(1, length(d_theta));
MSE_AML = zeros(1, length(d_theta));
MSE_DA = zeros(1, length(d_theta));
crb_sto = zeros(1, length(d_theta));
crb_sto_uc = zeros(1, length(d_theta));

for ii = 1:length(d_theta)
    
    theta_rad = [-2, -1, 0, 1, 2]*d_theta(ii)*pi/180;
    
    for run = 1:MC_runs 
        % Generate data
        [Y] = generate_ula_data(power_source, sig2, d, m, M, K, ...
               wavelength, theta_rad);
        
        % Estimate covariance matrix.
        [sample_cov, DA_out, AML_out, NML_out] =  estimates_cov(Y);

        DA_cov = DA_out.estimate;
        AML_cov = AML_out.estimate;
        NML_cov = NML_out.estimate;
        
        % Run root-music.
        doa_NML_cov = rmusic_1d(NML_cov, M, 2*pi*d/wavelength);
        doa_AML_cov = rmusic_1d(AML_cov, M, 2*pi*d/wavelength);
        doa_sample_cov = rmusic_1d(sample_cov, M, 2*pi*d/wavelength);
        doa_DA_cov = rmusic_1d(DA_cov, M, 2*pi*d/wavelength);
        
        % Compensate for different convention on angles.
        doa_NML_cov = sort(-doa_NML_cov.x_est);
        doa_AML_cov = sort(-doa_AML_cov.x_est);
        doa_sample_cov = sort(-doa_sample_cov.x_est);
        doa_DA_cov = sort(-doa_DA_cov.x_est);
        
        % Compute error
        MSE_SC(ii) = MSE_SC(ii) + norm(theta_rad - doa_sample_cov)^2;
        MSE_NML(ii) = MSE_NML(ii) + norm(theta_rad - doa_NML_cov)^2;
        MSE_AML(ii) = MSE_AML(ii) + norm(theta_rad - doa_AML_cov)^2;
        MSE_DA(ii) = MSE_DA(ii) + norm(theta_rad - doa_DA_cov)^2;
    end  
    MSE_SC(ii) = MSE_SC(ii)/(M*MC_runs);
    MSE_NML(ii) = MSE_NML(ii)/(M*MC_runs);
    MSE_AML(ii) = MSE_AML(ii)/(M*MC_runs);
    MSE_DA(ii) = MSE_DA(ii)/(M*MC_runs);  
    
    crb_sto(ii) = mean(diag(U_CRB(P, theta_rad, sig2, m, d, K)));
    crb_sto_uc(ii) = mean(diag(S_CRB(P, theta_rad, sig2, m, d, K)));
end
fprintf('\n');

%%
figure()
semilogy(d_theta, MSE_SC, '-x', d_theta, MSE_NML, '-x', ...
         d_theta, MSE_AML, '-x', d_theta, MSE_DA, '-x', ...
         d_theta, crb_sto, '--', d_theta, crb_sto_uc, '--');
ylabel('MSE', 'Interpreter', 'Latex', 'fontsize', 12); grid on;
legend('MSE_{SC}', 'MSE_{NML}', 'MSE_{AML}', 'MSE_{DA}', 'CRB', 'S-CRB');
xlabel('$\Delta \theta$', 'Interpreter', 'Latex')