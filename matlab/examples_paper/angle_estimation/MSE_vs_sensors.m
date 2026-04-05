% compares the performance of ULA with the CRB.
clear; clc;
setenv('BLAS_VERSION', '/usr/lib/x86_64-linux-gnu/libblas.so');
addpath('../algorithms/')
addpath('../utils')
%% Array geometry
theta_rad = [-10, -5, 0, 5, 10]*pi/180;
M = length(theta_rad);                        % number of sources
snr = 10;                                      % signal-to-noise ratio
power_source = 1;  
P = power_source*eye(M);                      % covariance matrix for source signals
sig2 = power_source*10^(-snr/10);             % noise variance
wavelength = 1;                               % normalized wavelength
d = wavelength/2;                             % spacing between sensors, in wavelength  

% Experiment parameters
sensors = (10:1:30);
MC_runs = 500;
K = 100;

% Containers for evaluating the performance
MSE_SC = zeros(1, length(sensors));
MSE_NML = zeros(1, length(sensors));
MSE_AML = zeros(1, length(sensors));
MSE_DA = zeros(1, length(sensors));
crb_sto = zeros(1, length(sensors));
crb_sto_uc = zeros(1, length(sensors));

DA_tot_time = 0;
AML_tot_time = 0;
NML_tot_time = 0;

for ii = 1:length(sensors)
    m = sensors(ii);
    fprintf("Simulating m = %i \n", m)
    
    for run = 1:MC_runs 
        % Generate data
        [Y] = generate_ula_data(power_source, sig2, d, m, M, K, ...
              wavelength, theta_rad);
        
        % Estimate covariance matrix.
        [sample_cov, DA_out, AML_out, NML_out] =  estimates_cov(Y);

        DA_cov = DA_out.estimate;
        AML_cov = AML_out.estimate;
        NML_cov = NML_out.estimate;
       
        DA_tot_time = DA_tot_time + DA_out.solve_time;
        AML_tot_time = AML_tot_time + AML_out.solve_time;
        NML_tot_time = NML_tot_time + NML_out.solve_time;
       
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
average_time_DA = DA_tot_time/(MC_runs*length(sensors));
average_time_AML = AML_tot_time/(MC_runs*length(sensors));
average_time_NML = NML_tot_time/(MC_runs*length(sensors));

fprintf('\n');

%%
figure()
semilogy(sensors, MSE_SC, '-x', sensors, MSE_NML, '-x', ...
         sensors, MSE_AML, '-x', sensors, MSE_DA, '-x', ...
         sensors, crb_sto, '--', sensors, crb_sto_uc, '--');
grid on;
ylabel('MSE', 'Interpreter', 'Latex', 'fontsize', 12); 
legend('SC', 'NML', 'AML', 'DA', 'U-CRB', 'S-CRB');
xlabel('$m$', 'Interpreter', 'Latex', 'fontsize', 12)
