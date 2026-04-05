clear; clc;
addpath('../../algorithms/NML')
addpath('../../algorithms/other_alg')
addpath('../../utils')
%%
all_n = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149];
number_of_runs = 50;
all_iter_NML1 = zeros(length(all_n), number_of_runs);
all_iter_NML2 = zeros(length(all_n), number_of_runs);

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
        S = 1/N*(X*X');
      
         % Newton's method.
        [out_NML1] = NML(S, X, 1);
        [out_NML2] = NML(S, X, 2);
        all_iter_NML1(sample_counter, run) = out_NML1.iter;
        all_iter_NML2(sample_counter, run) = out_NML2.iter;
    end 
end
max_iter_strategy1 = max(all_iter_NML1, [], 2);
max_iter_strategy2 = max(all_iter_NML2,[], 2);
average_iter_NML1 = sum(all_iter_NML1, 2)/number_of_runs;
average_iter_NML2 = sum(all_iter_NML2, 2)/number_of_runs;

%% Plot average number of Newton iterations for different initializations.
figure()
plot(all_n, average_iter_NML1', '-x', all_n, average_iter_NML2', '-x')
legend(["Strategy 1", "Strategy 2"], 'location', 'northwest'); grid on;
ylabel('Average number of iterations'); xlabel('$n$', 'Interpreter', 'Latex', 'fontsize', 15);
ylim([0, max(average_iter_NML1)])

