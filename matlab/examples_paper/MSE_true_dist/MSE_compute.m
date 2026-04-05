function [MSE_matrix, MSE_coefficients, num_of_PD_fails, num_of_unbounded, ...
         all_objs_ATOM, all_objs_NML, average_solve_times] = ...
    MSE_compute(methods, samples, true_cov, MC_runs)
% Computes the mean square error defined by (can change metric later, should be expectation?)
% MSE = norm('true_cov' - estimate, 'fro')^2. 
% Here estimate is the estimated covariance matrix from the methods defined
% 'methods'. The following methods are supported:
% 1. Sample covariance - SC
% 2. Newton's method   - NML
% 3. Asymptotic ML     - AML
% 4. Averaging along diagonals - AAD
% 5. ATOM

total_error_matrix_estimate = zeros(length(methods), length(samples));
total_error_coefficients = zeros(length(methods) - 1, length(samples));  % All but sample covariance.
all_objs_ATOM = zeros(length(samples), MC_runs);
all_objs_NML = zeros(length(samples), MC_runs);
average_solve_times = zeros(1, length(methods)-1);                         % All but sample covariance.
number_of_solve_times_counted = zeros(1, length(methods)-1);
num_of_PD_fails = zeros(length(methods), length(samples));
num_of_unbounded = zeros(1, length(samples));

if any(strcmp(methods, "AML"))
   [Psi_AML] = create_Psi_AML(size(true_cov, 1));
end

n = size(true_cov, 1) - 1;
first_row_real = real(true_cov(1, :));
first_row_imag = imag(true_cov(1, 2:end));


samples_counter = 0;
for N = samples 
    fprintf("Number of samples: %i \n", N)
    samples_counter = samples_counter + 1;
    for run = 1:MC_runs
        fprintf('N/run: %i / %i \n', N, run)
        X = generate_samples(true_cov, N);
        S = 1/N*(X*X');
        method_counter = 0;
        for method = methods
           method_counter = method_counter + 1;
           if method == "SC"
               total_error_matrix_estimate(method_counter, samples_counter) = ...
               total_error_matrix_estimate(method_counter, samples_counter)   ...
             + norm(true_cov - S, 'fro')^2;
           elseif method == "NML"
               [out_NML] = NML(S, X);
               total_error_matrix_estimate(method_counter, samples_counter) = ...
               total_error_matrix_estimate(method_counter, samples_counter)   ...
             + norm(true_cov - out_NML.estimate, 'fro')^2;
               % Assumes that SC is first so should subtract one one
               % method_counter.
               first_row_real_estimate = real(out_NML.estimate(1, :));
               first_row_imag_estimate = imag(out_NML.estimate(1, 2:end));
               
               total_error_coefficients(method_counter-1, samples_counter) =     ...
                   total_error_coefficients(method_counter-1, samples_counter)   ...
                +  norm(first_row_real - first_row_real_estimate)^2 ...
                +  norm(first_row_imag - first_row_imag_estimate)^2;
               if out_NML.unbounded == true
                  num_of_unbounded(1, samples_counter) = ... 
                      num_of_unbounded(1, samples_counter) + 1;
               end
               average_solve_times(method_counter - 1) = ...
                   average_solve_times(method_counter - 1) + out_NML.solve_time;
               number_of_solve_times_counted(method_counter - 1) = ...
                   number_of_solve_times_counted(method_counter - 1) + 1;
               all_objs_NML(samples_counter, run) = out_NML.ML_obj;
           elseif method == "NML_mex"
                




           end
           
           elseif method == "AML"
               if N >= n+1
                   [out_AML] = AML(S, N, Psi_AML);
                   total_error_matrix_estimate(method_counter, samples_counter) = ...
                      total_error_matrix_estimate(method_counter, samples_counter) ...
                    + norm(true_cov - out_AML.estimate, 'fro')^2;

                   first_row_real_estimate = real(out_AML.estimate(1, :));
                   first_row_imag_estimate = imag(out_AML.estimate(1, 2:end));

                   total_error_coefficients(method_counter-1, samples_counter) = ... 
                       total_error_coefficients(method_counter-1, samples_counter) ...
                    +  norm(first_row_real - first_row_real_estimate)^2 ...
                    +  norm(first_row_imag - first_row_imag_estimate)^2;
                   
                   average_solve_times(method_counter - 1) = ...
                       average_solve_times(method_counter - 1) + out_AML.solve_time;
                   number_of_solve_times_counted(method_counter - 1) = ...
                   number_of_solve_times_counted(method_counter - 1) + 1;
               end
           elseif method == "DA"
               tic;
               [estimate_AAD, first_row_real_estimate, first_row_imag_estimate] = AAD(S);
               average_solve_times(method_counter - 1) = ...
                   average_solve_times(method_counter - 1) + toc;
               number_of_solve_times_counted(method_counter - 1) = ...
                   number_of_solve_times_counted(method_counter - 1) + 1;
               %if min(eig(estimate)) <= 0
               %    num_of_PD_fails(method_counter, samples_counter) = ...
               %     num_of_PD_fails(method_counter, samples_counter) + 1;
               %end
               total_error_matrix_estimate(method_counter, samples_counter) = ...
                   total_error_matrix_estimate(method_counter, samples_counter) ...
                 + norm(true_cov - estimate_AAD, 'fro')^2;
               
                total_error_coefficients(method_counter-1, samples_counter) = ...
                   total_error_coefficients(method_counter-1, samples_counter) ...
                +  norm(first_row_real - transpose(first_row_real_estimate))^2 ...
                +  norm(first_row_imag - transpose(first_row_imag_estimate))^2;
                
           elseif method == "ATOM"
               gamma0 = 0.1;
               fprintf("ATOM starts. \n")
               [out_ATOM] = ATOM(S, gamma0, X);
               fprintf("ATOM finishes. \n")
               total_error_matrix_estimate(method_counter, samples_counter) = ...
                   total_error_matrix_estimate(method_counter, samples_counter) ...
                 + norm(true_cov - out_ATOM.estimate, 'fro')^2;
               first_row_real_estimate = real(out_ATOM.estimate(1, :));
               first_row_imag_estimate = imag(out_ATOM.estimate(1, 2:end));
               
               total_error_coefficients(method_counter-1, samples_counter) = ...
                   total_error_coefficients(method_counter-1, samples_counter) ...
                +  norm(first_row_real - first_row_real_estimate)^2 ...
                +  norm(first_row_imag - first_row_imag_estimate)^2;
            
                all_objs_ATOM(samples_counter, run) = out_ATOM.ML_obj;
                average_solve_times(method_counter - 1) = ...
                    average_solve_times(method_counter - 1) + out_ATOM.solve_time;
                number_of_solve_times_counted(method_counter - 1) = ...
                   number_of_solve_times_counted(method_counter - 1) + 1;
           end
        end
    end
end

average_solve_times = average_solve_times./number_of_solve_times_counted;
MSE_matrix = total_error_matrix_estimate/MC_runs;  
MSE_coefficients = total_error_coefficients/MC_runs;
end
