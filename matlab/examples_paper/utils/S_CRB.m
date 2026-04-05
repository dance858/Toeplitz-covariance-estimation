function [CRB] = S_CRB(P, theta_rad, sig2, m, d, N)
% From https://github.com/morriswmz/doa-tools
% Also see https://research.wmz.ninja/articles/2017/03/crbs-in-classical-doa-estimation-problems.html.

j=sqrt(-1);
A=exp(-2*pi*j*d*[0:m-1].'*sin(theta_rad));
DA = (-2*pi*j*d*[0:m-1].'*cos(theta_rad)).*exp(-2*pi*j*d*[0:m-1].'*sin(theta_rad));
p = diag(P);
noise_var = sig2;
snapshot_count = N;
k = length(theta_rad);


R = A * bsxfun(@times, p, A') + noise_var * eye(m);
R_inv = eye(m) / R;
R_inv = 0.5 * (R_inv + R_inv');
DRD = DA' * R_inv * DA;
DRA = DA' * R_inv * A;
ARD = A' * R_inv * DA;
ARA = A' * R_inv * A;
PP = p*p';
FIM_tt = 2*real((DRD.' .* ARA + conj(DRA) .* ARD) .* PP);   
FIM_pp = real(ARA.' .* ARA);                                % Similar to eq 8.189 in VanTrees.
R_inv2 = R_inv * R_inv;
FIM_ss = real(sum(diag(R_inv2)));                           % Similar to eq 8.194 in VanTrees.
% diag(AXB) = sum(A.' .* BX, 1);
FIM_tp = 2*real(conj(DRA) .* (bsxfun(@times, p, ARA)));
FIM_ts = 2*real(p .* sum(conj(DA) .* (R_inv2 * A), 1)');
FIM_ps = real(sum(conj(A) .* (R_inv2 * A), 1)');
FIM = [...
    FIM_tt  FIM_tp  FIM_ts; ...
    FIM_tp' FIM_pp  FIM_ps; ...
    FIM_ts' FIM_ps' FIM_ss] * snapshot_count;
CRB = eye(2*k+1) / FIM;
CRB = 0.5 * (CRB + CRB');
CRB = CRB(1:k, 1:k);
end