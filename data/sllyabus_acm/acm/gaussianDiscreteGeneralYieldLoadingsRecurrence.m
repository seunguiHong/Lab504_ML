function [By, Ay, b, a, Binf, Ainf] = gaussianDiscreteGeneralYieldLoadingsRecurrence(maturities, K0, K1, H0, rho0, rho1, delta0, delta1, timestep)
% function [By, Ay] = gaussianDiscreteYieldLoadingsRecurrence(maturities, K0d, K1d, H0d, rho0d, rho1d, timestep)
%
% K0      : N*1
% K1      : N*1
% H0      : N*N
% rho0    : scalar  
% rho1    : N*1
% delta0    : scalar  
% delta1    : N*1
% timestep : optional argument.
%
% By : N*M
% Ay : 1*M  (faster to not compute with only one output argument)
%
% r(t)   = rho0 + rho1'Xt
%        = 1 period discount rate
% d(t)   = delta0 + delta1'Xt
%        = 1 period dividend growth

% P(t)   = price of t-period zero coupon bond
%        = EQ0[exp(-r0 - r1 - ... - r(t-1)]
%        = exp(A+B'X0)
% yields = Ay + By'*X0
%   yield is express on a per period basis unless timestep is provided.
%   --For example, if the price of a two-year zero is exp(-2*.06)=exp(-24*.005),
%   --and we have a monthly model, the function will return Ay+By*X0=.005
%   --unless timestep=1/12 is provided in which case it returns Ay+By*X0=.06
%
% Where under Q:
% X(t+1) = K0 + K1*X(t) + eps(t+1),  cov(eps(t+1)) = H0
%
% A1 = delta0 - rho0 +. 5*delta1'*H0*delta1
% B1 = delta1 - rho1
% At = A(t-1) + K0'*B(t-1) .5*(delta1 + B(t-1))'*H0*(delta1 + B(t-1)) - rho0 + delta0
% Bt = B(t-1) + K1'*(delta1 + B(t-1)) - rho1d
%
% mautirities: 1*M # of periods

M = length(maturities);
N = length(K0);
Atemp = 0;
Btemp = zeros(N,1);
curr_mat = 1;
T = maturities(M);

if nargout > 2
    T = T + 60/timestep;
    Binf = zeros(N,T);
    Ainf = zeros(1,T);
end

for i = 1:T
    Atemp = Atemp + K0'*(delta1 + Btemp) +.5*(delta1 + Btemp)'*H0*(delta1 + Btemp) - rho0 + delta0;
    Btemp = K1'*(delta1 + Btemp) - rho1;
    if nargout > 2
        Ainf(i) = Atemp;
        Binf(:,i) = Btemp;
    end
    if curr_mat <= length(maturities)
        if i==maturities(curr_mat)
            Ay(1,curr_mat) = -Atemp/maturities(curr_mat);
            By(:,curr_mat) = -Btemp/maturities(curr_mat);
            b(:,curr_mat) = Btemp;
            a(:,curr_mat) = Atemp;
            curr_mat = curr_mat + 1;
        end
    end
end

if nargout > 4
    idx = 1:1/timestep:T;
    idx(1) = [];
    Ainf = Ainf(idx);
    Binf = Binf(:,idx);
end


if nargin==9
    Ay = Ay/timestep;
    By = By/timestep;
end
