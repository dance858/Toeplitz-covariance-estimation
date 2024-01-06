function [R_ATOM2, neg_log_lhood, obj_fun_equiv] = ATOM2(m, S, R0, MAX_ITERATION, K, gamma_0)
% DC CHANGED SAMPLE TO SAMPLE COVARIANCE S
%%
% sample - [y1, y2, ..., y_n]  complex matrix of size mxn
% R0 - init value for cov. matrix (e.g., sample covariance)
% K - works well with K = 5
% gamma - suggested using a value within [1e-4, 1e-1]
% MAX_ITERATION - suggested 500
if exist('FIXED_N_ITERATIONS','var') ~= 1
    FIXED_N_ITERATIONS = 0; % by default it is set to 0;
end


EPS_OUTER = 1e-4;
EPS_INNER = 1e-4;
MAX_IT_INNER = 20000;
MAX_IT_OUTER = MAX_ITERATION;
gamma = gamma_0; %1e-2; % hyperparameter of the algorithm,
% regulating its convergence speed

J = flip(eye(m,m));

%R_SCM = 1/n*(sample*sample');
R_SCM = S;

R_FB = 0.5*(R_SCM + J*conj(R_SCM)*J);

r = rank(R_FB);

%% INIZIALIZATION
[LLL,DDD] = ldl(R_FB);
DDD = DDD(1:r, 1:r);
LLL = LLL(:,1:r);

Y = LLL*(DDD^(1/2));

D = [[zeros(r,r), Y'];[Y, zeros(m,m)]];
R_t = R0;
E_t = blkdiag(Y'*pinv(R_t)*Y,R_t);
prec_R_t = R0;
%%
outer_it_count = 0;

neg_log_lhood = [];
neg_log_lhood(end+1) = real(trace(R_FB*pinv(R0)) + log(det(R_t)));
obj_fun_equiv = [];
obj_fun_equiv(end+1) = real(trace(Y'*pinv(R_t)*Y) + log(det(R_t)));

while(1) %% outer LOOP - MM
    outer_it_count = outer_it_count + 1;
    prev_g_inn_obj_fun = Inf;
    inner_it_count = 0;


    A_t = blkdiag(eye(size(D,1)-size(R_t,1)), pinv(R_t));
    B_t = E_t - (0.5/gamma)*A_t;
    T_k_t = B_t;
    P_k_t = zeros(size(T_k_t));
    Q_k_t = zeros(size(T_k_t));
    previous_R_k_t = T_k_t(end-m+1:end,end-m+1:end);
    while(1) %% Inner LOOP - Dykstra’s
        inner_it_count = inner_it_count + 1;
        Y_k_t = P_D_TOEP(T_k_t + P_k_t,r,m);   % step 1
        P_k_t = T_k_t + P_k_t - Y_k_t;         % step 2
        T_k_t = P_LMI(Y_k_t+Q_k_t, D);         % step 3
        Q_k_t = Y_k_t + Q_k_t - T_k_t;         % step 4;

        E_k_t = T_k_t;

        R_k_t = E_k_t(end-m+1:end,end-m+1:end);

        % 4) EXIT CONDITIONS INNER LOOP
        g_inn_obj_fun = real(trace(A_t*E_k_t)/gamma + trace(E_k_t^2)-2*trace(E_k_t*E_t));
        %         if(abs(prev_g_inn_obj_fun - g_inn_obj_fun)/abs(prev_g_inn_obj_fun) < EPS_INNER)
        if(norm(R_k_t - previous_R_k_t, 'fro')/norm(previous_R_k_t, 'fro') < EPS_INNER)
            break % EXIT INNER LOOP
        end
        %         if(inner_it_count == MAX_IT_INNER)
        %             break % EXIT INNER LOOP
        %         end
        prev_g_inn_obj_fun = g_inn_obj_fun;
        previous_R_k_t = R_k_t;
    end % END INNER LOOP

    E_t = E_k_t;
    R_t = E_t(end-m+1:end,end-m+1:end);
    X_t = E_t(1:r, 1:r);

    obj_fun_equiv(end+1) = real(trace(X_t) + log(det(R_t)));


    % EXIT CONDITIONS OUTER LOOP
    g_outer_obj_fun = real(trace(R_FB*pinv(R_t)) + log(det(R_t)));
    neg_log_lhood(end+1) = g_outer_obj_fun;

    gamma = gamma_0*(outer_it_count*log(outer_it_count+5))^K;


    if (not(FIXED_N_ITERATIONS))
        %         if(norm(R_t - prec_R_t,"fro")/norm(prec_R_t,"fro") < EPS_OUTER)
        if(abs(neg_log_lhood(end-1)-neg_log_lhood(end)) < EPS_OUTER)
            break % EXIT OUTER LOOP
        end
    end
    if(outer_it_count == MAX_IT_OUTER)
        break % EXIT OUTER LOOP
    end
    prec_R_t = R_t;
end % END MM
R_ATOM2 = R_t;
end

function PSI_PROJ = P_D_TOEP(PSI,r,m)
PSI_PROJ = zeros(size(PSI));

PSI_11 = PSI(1:r,1:r);
PSI_PROJ(1:r,1:r) = PSI_11;

PSI_22 = PSI(end-m+1:end,end-m+1:end);

p1 = zeros(1,m);
for i=1:m
    p1(i)=mean(diag(PSI_22,i-1));
end
PSI_PROJ(end-m+1:end,end-m+1:end) = toeplitz(p1);
end


function PSI_PROJ = P_LMI(PSI, D)
[V, U] = eig(PSI+D);
U = diag(real(U));
PSI_PROJ = V*diag(max(U,0))*V' - D;
end