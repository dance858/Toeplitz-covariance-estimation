clear; clc;
addpath('../algorithms/NML')
addpath('../algorithms/other_alg')
addpath('../utils')
%% 
methods = ["SC", "DA", "NML", "ATOM", "AML"]; % SC should always be first. AML last.
n = 10; samples = (20:20:200);
MC_runs = 5;

% Generate true covariance matrix
rng(0)
[true_cov] = toeplitz_via_cross_corr(n);



%%
[MSE_matrix, MSE_coefficients, num_of_PD_fails, unbounded, all_objs_ATOM, ...
    all_objs_NML, average_solve_time] = ...
    MSE_compute(methods, samples, true_cov, MC_runs);

%%
% Compute CRB
CRB = zeros(1, length(samples));
for k = 1:length(samples)
   N = samples(k); 
   FIM = CRB_true_dist(true_cov, N, n);
   CRB(k) = real(trace(inv(FIM)));
end


%% Figure 1 - MSE vs samples.
figure()
semilogy(samples, MSE_coefficients(1:end-1, :), '-x', ... 
         samples(3:end), MSE_coefficients(end, 3:end), '-x', ...
         samples, CRB, '--') 
xlabel('$N$', 'Interpreter', 'Latex')
ylabel('MSE', 'Interpreter', 'Latex', 'fontsize', 15)
legend([methods(2:end), 'CRB'], 'location', 'northeast')
grid on
ylim([0.5*min(CRB), 1.2*max(MSE_coefficients, [], 'all')])

%% Figure 1 - MSE vs samples but different ordering.
figure()
semilogy(samples, MSE_coefficients(2, :), '-x', ...
         samples(3:end), MSE_coefficients(4, 3:end), '-x', ... 
         samples, MSE_coefficients(1, :), '-x', ...
         samples, MSE_coefficients(3, :), '-x', ...
         samples, CRB, '--') 
xlabel('$K$', 'Interpreter', 'Latex')
ylabel('MSE', 'Interpreter', 'Latex', 'fontsize', 12)
grid on
legend('NML', 'AML', 'DA', 'ATOM', 'CRB', 'location', 'northeast')
ylim([0.5*min(CRB), 1.2*max(MSE_coefficients, [], 'all')])

%% Figure 2 - MSE vs samples but different ordering.
figure()
semilogy(samples(1:5), MSE_coefficients(2, 1:5), '-x', ... 
         samples(1:5), MSE_coefficients(1, 1:5), '-x', ...
         samples(1:5), MSE_coefficients(3, 1:5), '-x', ...
         samples(1:5), CRB(1:5), '--') 
xlabel('$N$', 'Interpreter', 'Latex')
ylabel('MSE', 'Interpreter', 'Latex', 'fontsize', 12)
grid on
legend('NML', 'DA', 'ATOM', 'CRB', 'location', 'northeast')
ylim([0.05, 1])

%% Figure 3
figure()
semilogy(samples(6:end), MSE_coefficients(2, 6:end), '-x', ... 
         samples(6:end), MSE_coefficients(1, 6:end), '-x', ...
         samples(6:end), MSE_coefficients(3, 6:end), '-x', ...
         samples(6:end), CRB(6:end), '--') 
xlabel('$N$', 'Interpreter', 'Latex')
ylabel('MSE', 'Interpreter', 'Latex', 'fontsize', 12)
grid on
legend('NML', 'DA', 'ATOM', 'CRB', 'location', 'northeast')
ylim([0.02, 0.1])


%%
semilogy(samples, MSE_coefficients, '-x', samples, CRB)
legend([methods(2:end), 'CRB'], 'location', 'northeast')
grid on
ylim([0.5*min(CRB), 1.2*max(MSE_coefficients, [], 'all')])

%% Figure 2 - MSE vs samples, normalized against CRB.
figure()
plot(samples, MSE_coefficients(1:end-1, :)./CRB, '-x') 
xlabel('$N$', 'Interpreter', 'Latex')
ylabel('MSE', 'fontsize', 15)
legend([methods(2:end-1)], 'location', 'northeast')
grid on

save("data/change_this_name.mat")