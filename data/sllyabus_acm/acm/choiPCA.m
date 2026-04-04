function [LAM,FAC,fitted_V,fitted_X,res_V,res_Y,SSE_V,SSE_Y,SS,WSSE_V,WSSE_Y,WSSE] = choiPCA(obs,J1,flagSV,varargin)
% the state space is drvien by z where z = (x',v')'. 
% J observables in total, but the last Jm observables are driven only by m-factors
[J,T] = size(obs);
J2 = J-J1;
S = [zeros(J1,J2); eye(J2)];
if isempty(varargin)
    sigma_e = speye(J);
else
    sigma_e = varargin{1};
%     sigma_e(J1+1:end,J1+1:end) = diag(diag(sigma_e(J1+1:end,J1+1:end)));
%     sigma_e = blkdiag(sigma_e(1:J1,1:J1),diag(diag(sigma_e(J1+1:end,J1+1:end))));
end
K = length(flagSV);
res = obs;
FAC = [];
LAM = [];

for i = 1:K
    isSV = flagSV(i);
    if isSV
        % get factor
        target = res'/sigma_e*res;
        [fac, a] = eigs(target,1);
        if a < 0
            fac = -fac;
        end


        % normalize factor
        sfac = sqrt(mean(fac.^2));
        V = fac'./sfac;

        % get loadings        
        psi = res*V'/(V*V');
        if all(psi(J1)<0)
            lam = -psi;
            V = -V;
        else
            lam = psi;
        end
        
        % get residuals
        FAC = [FAC;V];
        LAM = [LAM,lam];
        res = obs - LAM*FAC;
    else
        % get factor
        S_e = S/(S'/sigma_e*S)*S'/sigma_e;
        target = res'*(2*eye(J)-S_e')/sigma_e*S_e*res;
        [fac, a] = eigs(target,1);
        if a < 0
            fac = -fac;
        end

        % normalize factor
        sfac = sqrt(mean(fac.^2));
        X = fac'./sfac;

        % get loadings
        psi = (S'/sigma_e*S)\S'/sigma_e*res*X'/(X*X');
        lam = S*psi;
        
        % get residuals
        FAC = [FAC;X];
        LAM = [LAM,lam];
        res = obs - LAM*FAC;
    end
end
fitted = LAM*FAC;
fitted_V = fitted(1:J1,:);
fitted_X = fitted(J1+1:end,:);
res = obs - fitted;
res_V = obs(1:J1,:) - fitted_V;
res_Y = obs(J1+1:end,:) - fitted_X;
SSE_V = sum(sum(res_V.^2))/(T*J1);
SSE_Y = sum(sum(res_Y.^2))/(T*J2);


% SS = diag(diag(res*res'/T));
SS = res*res'/T;
if isempty(varargin)
    WSSE = trace(res'*res)/(T*J);
    WSSE_V = trace(res_V'*res_V)/(T*J1);
    WSSE_Y = trace(res_Y'*res_Y)/(T*J2);
else
    WSSE = trace(res'/sigma_e*res)/(T*J);
    WSSE_V = trace(res_V'/sigma_e(1:J1,1:J1)*res_V)/(T*J1);
    WSSE_Y = trace(res_Y'/sigma_e(J1+1:end,J1+1:end)*res_Y)/(T*J2);
end


% SSE2 = vec(res)'*kron(eye(T),inv(sigma_e))*vec(res);

% 
% function [fac,lam,eigval,ssr] = pc_factor(x,k)
% %    Compute principal components estimates of factor model
% %
% %    Input:
% %     x = txn matrix
% %     k = number of factors
% %
% %     Model
% %
% %     x(it) = lam(i)'f(t) + u(it)
% %
% %     or
% %
% %     x = f*lam' + u
% %
% %
% %     Output:
% %
% %     f = txk matrix of factors
% %     lam = n*k matrix of factor loadings
% %     eigval = kx1 vector of eigenvalues of x*x' (or x'x) ...
% %     ssr = sum of squared u's
% %
% %     Normalization:
% %      f is normalized so that each column has std dev = 1, thus F'F = t*I(k)
% %
% %     Calculation note:
% %      Calculations are speeded by using the smaller of x*x' or x'x
% [t,n]=size(x);
% 
% if k > 0
%     if n < t
%         xx=x'*x;
%         [ve,va]=eig(xx);
%         va=diag(va);
% 
%         va=flipud(va); %sort eigen value from large to small.
%         ve=fliplr(ve); %sort the corresponding eigen vectors.
%         eigval=va(1:k); %keep the first k eigen value.
%         lam=ve(:,1:k); %keep the first k eigen vectors.
%         fac=x*lam;
%     else
%         xx=x*x';
%         [ve,va]=eig(xx);
%         va=diag(va);
%         
%         va=flipud(va); %sort eigen value from large to small.
%         ve=fliplr(ve); %sort the corresponding eigen vectors.
%         eigval=va(1:k);
%         fac=ve(:,1:k);
%     end
%     %@ Normalize @
%     sfac=sqrt(mean(fac.^2));
%     fac=fac./sfac;
%     lam=(x'*fac)/t;  % Note fac'fac = t @
%     ssr=sum(va)-sum(eigval);
% else
%     ssr=sum(sum(x.^2));
%     lam=nan;
%     eigval=nan;
%     fac=nan;
%    end
% end
% end
