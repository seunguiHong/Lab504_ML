function [Ehprx, Z_t] = ACM_Erx12(N, yyyymm, yields, spec)

[T, J] = size(yields);

dt = 1/12;                                                % monthly step (years)

% =========================================================================
% 2) Log prices and 1m excess holding-period log returns
% =========================================================================
t = 1:(T-1);
n = 1:(J-1);

p = -(dt) * yields .* (1:J);                              % logP(t,m) = -(m/12)*y_t^(m)

hpr = NaN(T, J-1);                                        % hpr(t+1,n) = logP(t+1,n) - logP(t,n+1)
hpr(t+1, n) = p(t+1, n) - p(t, n+1);

br1  = [NaN; dt * yields(t, 1)];                          % rf_1m(t+1) = (1/12)*y_t^(1m)
hprx = hpr - br1;                                         % excess vs 1m rf

% =========================================================================
% 3) PCA factors and ACM inputs
% =========================================================================
ymats   = 3:120;                                        % PCA basis maturities (months)
rmats_m = [6:6:60, 84, 120];                              % maturities used in ACM estimation (months)
rmats_i = rmats_m - 1;                                    % columns in hprx (returns are 1m..119m)

X = demean(yields(:, ymats))';                            % |ymats|-by-T
[~, FAC] = choiPCA(X, 0, zeros(N, 1));                    % FAC: N-by-T

FAC(2, :) = -FAC(2, :);
FAC(3, :) = -FAC(3, :);
FAC(5, :) = -FAC(5, :);

Z  = FAC;                                                 % N-by-T
rx = hprx(:, rmats_i)';                                   % |rmats_m|-by-T

% =========================================================================
% 4) ACM estimation (Q-dynamics)
% =========================================================================
[K0, K1, Sigma, R0, B, R1, lambda0, Lambda1, ~, ~] = ACMQ(Z, rx, spec); %#ok<ASGLU>

K0Q = K0 - lambda0;
K1Q = K1 - Lambda1;

% =========================================================================
% 5) Model-implied expected holding-period returns (12m holding period)
% =========================================================================
rf_row = br1(t+1)';                                       % 1-by-(T-1)
Z_lag  = [ones(1, T-1); Z(:, t)];                         % (1+N)-by-(T-1)

rf_map = rf_row / Z_lag;                                  % rf = r0 + r1' * Z_t (as coded)
r0     = rf_map(1);
r1     = rf_map(2:end)';

[By, Ay] = gaussianDiscreteGeneralYieldLoadingsRecurrence( ...
    1:120, K0Q, K1Q, Sigma, r0, r1, 0, zeros(N, 1), dt);

H = 1/dt;                                                 % 12 months
m = (1:J)';                                               % 1..J (months)

[temp0, temp1] = fcast_ar1(H, K0, K1);
Z_tPlusH = temp0 + temp1*Z(:,end);
fcasted  = Ay' + By' * Z_tPlusH ;                        % forecast yields at t+H (J-by-1)
observed = yields(end, :)';                               % observed yields at t (J-by-1)

p_tPlusH = -(dt) * fcasted  .* m;                         % logP_{t+H}(m)
p_t      = -(dt) * observed .* m;                            % logP_t(m)

Ehpr = NaN(J, 1);
Ehpr(H+1:J) = p_tPlusH(1:J-H) - p_t(H+1:J);               % p_{t+H}(m-H) - p_t(m)

Ehprx = NaN(J, 1);
Ehprx(H+1:J) = Ehpr(H+1:J) - yields(end, H);              % subtract 1y yield (12m)

Z_t = Z(:, end);

if spec.VERBOSE
    dates  = datetime(lbusdate(floor(yyyymm/100), mod(yyyymm,100)),"ConvertFrom","datenum"); 
    LineWidth = 2;
    FontSize = 16;
    idx = [24;120];
    [~,pos] = intersect([1:120]',idx);
    observed = yields';
    fitted  = Ay' + By'*Z;                                       % T-by-120 model-implied yields
    figure;
    for i = 1:length(idx)
        subplot(1,2,i);
        plot(dates, 100*observed(pos(i),:)','-b','LineWidth',LineWidth)
        hold on
        plot(dates, 100*fitted(pos(i),:)','--g','LineWidth',LineWidth)
        hold on
        axis tight
        ylabel('Percent')
        legend({'Observed','Fitted'})
        xtickformat('yyyy')
        set(gca, 'FontName', 'Times New Roman', 'FontSize',FontSize )
        title([num2str(idx(i)*dt), '-year yield'])
    end
end

