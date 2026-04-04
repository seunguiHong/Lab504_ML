function [K0,K1,Sigma,R0,B,R1,lambda0,Lambda1,residual_factor,residual_return] = ACMQ(Z,rx,spec)

%--------------------------------------------------------------------------------------------
% Z:    (N by T) factors
% ret:  (J by T) log return or log excess return (unannualized)
%--------------------------------------------------------------------------------------------

%% 0. some new variables for convenience
[N,T] = size(Z);
[J,~] = size(rx);
t     = 1:T-1;

%% 1. factor dynamics
if spec.bc == 1
    [K1, K0, Sigma] = est_bc_var(Z', 1, 1, 1, 5000, 1000, 5, 0);
    residual_factor  = Z(:,t+1) - (K0+K1*Z(:,t));
else
    K = mrdivide(Z(:,t+1),[ones(1,T-1);Z(:,t)]);
    K0 = K(:,1);
    K1 = K(:,2:end);
    residual_factor = Z(:,t+1) - K*[ones(1,T-1);Z(:,t)];
    Sigma  = residual_factor * residual_factor'/(T-1);
end

%% 2. excess-return regression
lhv  = rx(:,t+1);
rhv  = [ones(1,T-1);Z(:,t+1);Z(:,t)];
temp = mrdivide(lhv,rhv);
residual_return = lhv - temp*rhv;

R0   = temp(:,1);
B    = temp(:,2:2+N-1);
R1   = temp(:,end-N+1:end);

% obtain C based on Sigma and B
C0 = nan(J,1);
for i = 1:J
    C0(i) = .5*B(i,:)*Sigma*B(i,:)';
end

%% 3. solve for K0Q and K1Q
K1Q = mldivide(-B,R1);
K0Q = -B\(R0 + C0);

lambda0 = K0 - K0Q;
Lambda1 = K1 - K1Q;
