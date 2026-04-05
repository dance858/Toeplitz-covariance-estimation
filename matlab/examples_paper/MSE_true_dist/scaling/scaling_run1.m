clear; clc;
addpath('../../algorithms/NML')
addpath('../../algorithms/other_alg')
addpath('../../utils')
%%
init_strategy = 1;
all_n = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104];
number_of_runs = 10;
gamma = 0.1;

solve_times_ATOM = zeros(1, length(all_n));
solve_times_NML = zeros(1, length(all_n));
solve_times_NML_mex = zeros(1, length(all_n));
solve_times_AML = zeros(1, length(all_n));
solve_times_DA = zeros(1, length(all_n));

all_objs_ATOM = zeros(length(all_n), number_of_runs);
all_objs_NML = zeros(length(all_n), number_of_runs);


sample_counter = 0;
for n = all_n
    fprintf("n: %i \n", n)
    sample_counter = sample_counter + 1;
    [true_cov] = toeplitz_via_cross_corr(n);
    N = (n+1);
    for run = 1:number_of_runs
        if rem(run, 5) == 0
           fprintf("Run: %i \n", run) 
        end
        X = generate_samples(true_cov, N);
       
      
         % Estimate covariance matrix.
        [sample_cov, DA_out, AML_out, NML_out, NML_mex_out] =  estimates_cov(X);
    
        solve_times_DA(1, sample_counter) = solve_times_DA(1, sample_counter) + DA_out.solve_time;
        solve_times_NML(1, sample_counter) = solve_times_NML(1, sample_counter) + NML_out.solve_time;
        solve_times_NML_mex(1, sample_counter) = solve_times_NML_mex(1, sample_counter) + NML_mex_out.solve_time;
        solve_times_AML(1, sample_counter) = solve_times_AML(1, sample_counter) + AML_out.solve_time;

        % ATOM
        S = 1/N*(X*X');
        [out_ATOM] = ATOM(S, gamma, X);
        all_objs_ATOM(sample_counter, run) = out_ATOM.ML_obj;
        solve_times_ATOM(1, sample_counter) = ...
        solve_times_ATOM(1, sample_counter) + out_ATOM.solve_time;
    
    
    
    end 
end

solve_times_NML = solve_times_NML/number_of_runs;
solve_times_NML_mex = solve_times_NML_mex/number_of_runs;
solve_times_ATOM = solve_times_ATOM/number_of_runs;
solve_times_AML = solve_times_AML/number_of_runs;
solve_times_DA = solve_times_DA/number_of_runs;



%% Plot that shows average solve time.
figure()
semilogy(all_n + 1, solve_times_NML_mex, '-x', all_n + 1, solve_times_AML, '-x', ...
        all_n + 1, solve_times_DA, '-x', all_n + 1, solve_times_ATOM, '-x');
legend(["NML", "AML", "DA", "ATOM"], 'location', 'northwest'); grid on;
ylabel('Solve time (s)', 'Interpreter', 'Latex', 'fontsize', 12); xlabel('$n + 1$', 'Interpreter', 'Latex', 'fontsize', 12);
xlim([0, 110]);

%%
figure()
semilogy(all_n + 1, solve_times_NML_mex, '-x', all_n + 1, solve_times_AML, '-x', ...
         all_n + 1, solve_times_ATOM, '-x');
legend(["NML", "AML", "ATOM"], 'location', 'northwest'); grid on;
ylabel('Solve time (s)', 'Interpreter', 'Latex', 'fontsize', 12); xlabel('$n + 1$', 'Interpreter', 'Latex', 'fontsize', 12);
xlim([0, 110]);


%% Plot that illustrates achieved objective value.
figure()
plot(all_n, mean(all_objs_ATOM./all_objs_NML, 2), 'o')