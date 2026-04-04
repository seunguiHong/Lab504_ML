function [Phi_tilde, mu_tilde, V_tilde] = est_bc_var(X, ev_restr, p, flag_mean, N, N_burn, B, check, B_check)
% est_bc_var - bias-corrected VAR estimation using stochastic approximation (SA)
% see Bauer, Rudebusch, Wu (2012, JBES) Correcting Estimation Bias in Dynamic
% Term Structure Models
% inputs:
%  X       REQUIRED. data matrix, T x k
%  ev_restr  largest eigenvalue not to be exceeded by Phi_tilde
%  p       order of the VAR, default VAR(1), p=2 possible, p>2 not implemented
%  flag_mean     flag whether mean- (TRUE) or median- (FALSE) unbiased estimation is desired
%           default TRUE (mean-unbiased)
%  N       number of iterations of the SA algorithm after burn-in (default 10,000)
%  N_burn  number of burn-in iterations (default 100)
%  B       number of bootstrap samples to calculate noisy measure of mean/median
%          of the OLS estimator (default 50)
%  check   flag whether closeness check is to be performed in the end (default TRUE)
%  B_check number of bootstrap samples for closeness check (default
%  100,000)
%
% outputs:
%  Phi_tilde  mean-reversion matrix
%  mu_tilde   intercept
%  V_tilde    variance-covariance matrix of residuals

% parameters
if nargin<9; B_check = 100000; end;
if nargin<8; check = true; end;
if nargin<7; B = 10; end;
if nargin<6; N_burn = 1000; end;
if nargin<5; N = 5000; end;
if nargin<4; flag_mean = true; end;
if nargin<3; p = 1; end;
if nargin<2; ev_restr = 1; end;
gamma_i = .2;

if (flag_mean)
    fprintf('Mean-');
else
    fprintf('Median-');
end
fprintf('Bias-adjustment for VAR estimates\n');
fprintf('N = %u, N_burn = %u, B = %u, B_check = %u, p = %u\n', [N,N_burn,B,B_check,p]);

[T,k] = size(X);

if (p==1)
    % first-order VAR
    
    % OLS
    [Phi_hat] = est_var(X, 1, true, false);
    fprintf('largest absolute eigenvalue OLS:  %8.6f \n', max(abs(eig(Phi_hat))));
    
    % initialization for SA algorithm
    theta = zeros(k^2, N_burn+N);
    theta_hat = Phi_hat(:);
    theta(:,1) = theta_hat; % starting value
    
    % SA algorithm
    for j=1:N_burn+N-1
        Phi_new = m_var(theta(:,j), B, 1, X, flag_mean);
        theta_new = Phi_new(:);
        d = theta_hat - theta_new;
        theta(:,j+1) = theta(:,j) + gamma_i*d;
        %if ( (j>N_burn) && (mod(j-N_burn, 100) == 0) )
        %    fprintf('****** iteration %u ******\n', j-N_burn);
        %    theta_tilde = mean(theta(:,N_burn+1:j),2);
        %    Phi_tilde = reshape(theta_tilde,k,k);
        %    fprintf('largest absolute eigenvalue:  %8.6f \n', max(abs(eig(Phi_tilde))));
        %end
    end
    
    theta_tilde = mean(theta(:,(N_burn+1):(N_burn+N)),2);
    
    if (check)
        % check whether mean/median of OLS---given that DGP is VAR with Phi_tilde---is close to actual OLS estimates
        disp('... checking closeness of mean/median to actual estimates ...');
        [Phi_new] = m_var(theta_tilde, B_check, 1, X, flag_mean);
        theta_new = Phi_new(:);
        dist = sqrt(sum((theta_new - theta_hat).^2)/length(theta_new));
        fprintf('root mean square distance: %8.6f \n', dist);
    end
    
    % bias_adjusted estimates
    Phi_tilde = reshape(theta_tilde,k,k);
    ev_bc = max(abs(eig(Phi_tilde)));   % get scalar largest absolute eigenvalues
    fprintf('largest eigenvalue after BC:  %8.6f \n', ev_bc);
    
    % impose restriction on eigenvalue
    Phi_tilde = shrink_phi(Phi_tilde, Phi_hat, ev_restr);
    
    % choose intercept to match sample mean
    mu_tilde = (eye(k) - Phi_tilde) * mean(X)';
    
    % residuals and their variance-covariance matrix
    Xdem = X - ones(T,1)*mean(X);
    resid_tilde = Xdem(2:T,:)' - Phi_tilde * Xdem(1:(T-1),:)';
    V_tilde = resid_tilde * resid_tilde'/(T-1);
else
    error('bias-corrected VAR estimation only implemented for first-order VAR');
end


end