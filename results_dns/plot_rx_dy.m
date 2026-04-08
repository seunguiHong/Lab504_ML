clear; clc; close all;

%% =========================================================================
% User settings
% =========================================================================

target_n   = 10;                 % draws dy_{n-1} and rx^{(n)}
res_file   = 'dns_oos_compat_dns_yield_h12.mat';         % result .mat file
csv_file   = 'dataset.csv';      % raw yield csv
model_name = 'DNSModel';          % e.g. 'NNModel'

if ~isscalar(target_n) || target_n ~= floor(target_n) || target_n < 2
    error('target_n must be an integer >= 2.');
end

dy_idx = target_n - 1;
dy_name = "dy_" + string(dy_idx);

eval_start_dy = datetime(1989,12,31);
eval_end_dy   = datetime(2024,12,31);

eval_start_rx = datetime(1989,12,31);
eval_end_rx   = datetime(2023,12,31);

eh_start   = datetime(1971,8,31);
hac_lags   = 12;
min_obs_cs = 12;

fig_main = [100, 100, 1200, 520];
fig_sse  = [120, 140, 1200, 430];

lw_true  = 2.4;
lw_model = 1.8;
lw_bench = 1.8;
lw_sse   = 2.0;

fs_axis   = 18;
fs_legend = 15;
fs_tick   = 13;

%% =========================================================================
% Load result file
% =========================================================================

S = load(res_file);

if ~isfield(S, 'Dates')
    error('Dates is missing in result file.');
end
if ~isfield(S, 'Y_Columns')
    error('Y_Columns is missing in result file.');
end
if ~isfield(S, 'Y_True')
    error('Y_True is missing in result file.');
end

agg_field = "Y_forecast_agg_" + model_name;
if ~isfield(S, agg_field)
    error('Missing field %s in result file.', agg_field);
end

if isdatetime(S.Dates)
    dates_res = S.Dates(:);
elseif isstring(S.Dates)
    dates_res = datetime(S.Dates(:));
elseif iscell(S.Dates)
    dates_res = datetime(string(S.Dates(:)));
elseif ischar(S.Dates)
    dates_res = datetime(string(cellstr(S.Dates)));
else
    error('Unsupported Dates format.');
end

if isstring(S.Y_Columns)
    ycols = S.Y_Columns(:);
elseif iscell(S.Y_Columns)
    ycols = string(S.Y_Columns(:));
elseif ischar(S.Y_Columns)
    ycols = string(cellstr(S.Y_Columns));
else
    error('Unsupported Y_Columns format.');
end

Y_true = S.Y_True;
Y_hat  = S.(agg_field);

if size(Y_true,1) ~= numel(dates_res) || size(Y_hat,1) ~= numel(dates_res)
    error('Dates and forecast arrays are inconsistent.');
end

j_dy = find(ycols == dy_name, 1);
if isempty(j_dy)
    error('Missing %s in Y_Columns.', dy_name);
end

dy_true_full = Y_true(:, j_dy);
dy_hat_full  = Y_hat(:, j_dy);

%% =========================================================================
% dy_{n-1}: realized vs model-implied
% =========================================================================

dy_rw_full = zeros(size(dy_true_full));

ok = isfinite(dy_true_full) & isfinite(dy_hat_full) & isfinite(dy_rw_full);
dates_dy = dates_res(ok);
dy_true  = dy_true_full(ok);
dy_hat   = dy_hat_full(ok);
dy_rw    = dy_rw_full(ok);

ok_eval = (dates_dy >= eval_start_dy) & (dates_dy <= eval_end_dy);
dates_dy = dates_dy(ok_eval);
dy_true  = dy_true(ok_eval);
dy_hat   = dy_hat(ok_eval);
dy_rw    = dy_rw(ok_eval);

if isempty(dates_dy)
    error('No dy observations remain in evaluation window.');
end

mse_dy   = mean((dy_true - dy_hat).^2);
rmse_dy  = sqrt(mse_dy);
mae_dy   = mean(abs(dy_true - dy_hat));
corr_dy  = corr(dy_true, dy_hat, 'Rows', 'complete');

ss_res_dy = sum((dy_true - dy_hat).^2);
ss_bmk_dy = sum(dy_true.^2);
r2oos_dy  = 1 - ss_res_dy / ss_bmk_dy;

valid = isfinite(dy_true) & isfinite(dy_hat);
if sum(valid) >= 2
    yt = dy_true(valid);
    yf = dy_hat(valid);
    y0 = zeros(size(yt));
    f  = (yt - y0).^2 - (yt - yf).^2 + (y0 - yf).^2;
    Xc = ones(numel(f),1);
    try
        [~,~,stats] = hac(Xc, f, 'Intercept', false, 'Lags', hac_lags);
        tval = stats.beta ./ stats.se;
    catch
        fit = fitlm(Xc, f, 'Intercept', false);
        tval = fit.Coefficients.tStat(1);
    end
    pval_dy = 1 - tcdf(tval, numel(f) - 1);
else
    pval_dy = NaN;
end

sq_err_model = (dy_true - dy_hat).^2;
sq_err_zero  = dy_true.^2;
scale_den = (numel(dy_true) - 1) * var(dy_true, 0);
cum_sse_diff = cumsum((sq_err_zero - sq_err_model) / scale_den);

fprintf('\n');
fprintf('dy summary\n');
fprintf('Target     : %s\n', dy_name);
fprintf('N          : %d\n', numel(dy_true));
fprintf('MSE        : %.6f\n', mse_dy);
fprintf('RMSE       : %.6f\n', rmse_dy);
fprintf('MAE        : %.6f\n', mae_dy);
fprintf('Corr       : %.6f\n', corr_dy);
fprintf('R2OOS      : %.6f\n', r2oos_dy);
fprintf('p-value    : %.6f\n', pval_dy);

dy_label = sprintf('$\\Delta y^{(%d)}$', dy_idx);

figure('Color','w','Position',fig_main);
plot(dates_dy, dy_true, '--', 'LineWidth', lw_true); hold on;
plot(dates_dy, dy_hat,  '-',  'LineWidth', lw_model);
plot(dates_dy, dy_rw,   '-',  'LineWidth', lw_bench);

recessionplot('axes', gca);
uistack(findobj(gca,'Tag','recessionbars'),'bottom');

grid on; box on;
ax = gca; ax.FontSize = fs_tick; ax.LineWidth = 1.1;
xlim([dates_dy(1), dates_dy(end)]);
xlabel('Date', 'FontSize', fs_axis, 'FontWeight', 'bold');
ylabel(dy_label, 'Interpreter', 'latex', 'FontSize', fs_axis, 'FontWeight', 'bold');

lgd = legend("Realized", "Model-implied", "RW (0)", 'Location', 'best');
lgd.FontSize = fs_legend;
lgd.Box = 'off';

figure('Color','w','Position',fig_sse);
plot(dates_dy, cum_sse_diff, '-', 'LineWidth', lw_sse); hold on;
yline(0, '--');

recessionplot('axes', gca);
uistack(findobj(gca,'Tag','recessionbars'),'bottom');

grid on; box on;
ax = gca; ax.FontSize = fs_tick; ax.LineWidth = 1.1;
xlim([dates_dy(1), dates_dy(end)]);
xlabel('Date', 'FontSize', fs_axis, 'FontWeight', 'bold');
ylabel('Cumulative SSE difference', 'FontSize', fs_axis, 'FontWeight', 'bold');

%% =========================================================================
% rx^{(n)} reconstruction
% =========================================================================

TBL = readtable(csv_file);

m_prev = sprintf('m%03d', 12 * (target_n - 1));
m_curr = sprintf('m%03d', 12 * target_n);
req = {'Time', 'm012', m_prev, m_curr};

if ~all(ismember(req, TBL.Properties.VariableNames))
    error('CSV must contain %s.', strjoin(req, ', '));
end

dates_csv = datetime(string(TBL.Time), 'InputFormat', 'yyyyMM') + calmonths(1) - days(1);
y1_full   = TBL.m012(:);
y_nm1_full = TBL.(m_prev)(:);
y_n_full   = TBL.(m_curr)(:);

y_nm1_tp12_full = [y_nm1_full(13:end); NaN(12,1)];
rx_true_full = target_n .* y_n_full - (target_n - 1) .* y_nm1_tp12_full - y1_full;

rx_eh_full = NaN(size(rx_true_full));
eh_start_idx = find(dates_csv == eh_start, 1, 'first');
if isempty(eh_start_idx)
    error('EH start date is not found in csv_file.');
end

for t = (eh_start_idx + 1):numel(rx_true_full)
    hist_sample = rx_true_full(eh_start_idx:t-1);
    rx_eh_full(t) = mean(hist_sample, 'omitnan');
end

[dates_rx, ia, ib] = intersect(dates_res, dates_csv, 'stable');

dy_hat_match = dy_hat_full(ia);
y1   = y1_full(ib);
y_nm1 = y_nm1_full(ib);
y_n   = y_n_full(ib);
rx_true = rx_true_full(ib);
rx_eh   = rx_eh_full(ib);

y_nm1_hat_tp12 = y_nm1 + dy_hat_match;
rx_model = target_n .* y_n - (target_n - 1) .* y_nm1_hat_tp12 - y1;
rx_rw    = target_n .* y_n - (target_n - 1) .* y_nm1 - y1;

slope_n = y_n - y1;
T = numel(rx_true);
rx_cs = NaN(T,1);

for t = 1:T
    train_end = t - hac_lags;
    if train_end < 2
        continue;
    end

    y_reg = rx_true(1:train_end);
    x_reg = slope_n(1:train_end);

    ok = isfinite(y_reg) & isfinite(x_reg);
    if sum(ok) < min_obs_cs
        continue;
    end

    Xreg = [ones(sum(ok),1), x_reg(ok)];
    b = Xreg \ y_reg(ok);

    if isfinite(slope_n(t))
        rx_cs(t) = [1, slope_n(t)] * b;
    end
end

ok = isfinite(rx_true) & isfinite(rx_model) & isfinite(rx_rw) & isfinite(rx_cs) & isfinite(rx_eh);
dates_rx = dates_rx(ok);
rx_true  = rx_true(ok);
rx_model = rx_model(ok);
rx_rw    = rx_rw(ok);
rx_cs    = rx_cs(ok);
rx_eh    = rx_eh(ok);

ok_eval = (dates_rx >= eval_start_rx) & (dates_rx <= eval_end_rx);
dates_rx = dates_rx(ok_eval);
rx_true  = rx_true(ok_eval);
rx_model = rx_model(ok_eval);
rx_rw    = rx_rw(ok_eval);
rx_cs    = rx_cs(ok_eval);
rx_eh    = rx_eh(ok_eval);

if isempty(dates_rx)
    error('No rx observations remain in evaluation window.');
end

ss_eh = sum((rx_true - rx_eh).^2);

r2_model = 1 - sum((rx_true - rx_model).^2) / ss_eh;
r2_rw    = 1 - sum((rx_true - rx_rw   ).^2) / ss_eh;
r2_cs    = 1 - sum((rx_true - rx_cs   ).^2) / ss_eh;
r2_eh    = 1 - sum((rx_true - rx_eh   ).^2) / ss_eh;

cases = {'MODEL','RW','CS'};
yhats = {rx_model, rx_rw, rx_cs};
pvals = NaN(3,1);

for i = 1:3
    valid = isfinite(rx_true) & isfinite(yhats{i}) & isfinite(rx_eh);
    if sum(valid) >= 2
        yt = rx_true(valid);
        yf = yhats{i}(valid);
        y0 = rx_eh(valid);
        f  = (yt - y0).^2 - (yt - yf).^2 + (y0 - yf).^2;
        Xc = ones(numel(f),1);
        try
            [~,~,stats] = hac(Xc, f, 'Intercept', false, 'Lags', hac_lags);
            tval = stats.beta ./ stats.se;
        catch
            fit = fitlm(Xc, f, 'Intercept', false);
            tval = fit.Coefficients.tStat(1);
        end
        pvals(i) = 1 - tcdf(tval, numel(f) - 1);
    end
end

fprintf('\n');
fprintf('rx summary\n');
fprintf('Target     : rx^(%d)\n', target_n);
fprintf('R2 MODEL   : %.6f\n', r2_model);
fprintf('R2 RW      : %.6f\n', r2_rw);
fprintf('R2 CS      : %.6f\n', r2_cs);
fprintf('R2 EH      : %.6f\n', r2_eh);
fprintf('p MODEL    : %.6f\n', pvals(1));
fprintf('p RW       : %.6f\n', pvals(2));
fprintf('p CS       : %.6f\n', pvals(3));

figure('Color','w','Position',fig_main);
plot(dates_rx, rx_true,  '--', 'LineWidth', lw_true); hold on;
plot(dates_rx, rx_model, '-',  'LineWidth', lw_model);
plot(dates_rx, rx_rw,    '-',  'LineWidth', lw_bench);
plot(dates_rx, rx_cs,    '-',  'LineWidth', lw_bench);
plot(dates_rx, rx_eh,    '--', 'LineWidth', lw_bench);

recessionplot('axes', gca);
uistack(findobj(gca,'Tag','recessionbars'),'bottom');

grid on; box on;
ax = gca; ax.FontSize = fs_tick; ax.LineWidth = 1.1;
xlim([dates_rx(1), dates_rx(end)]);
xlabel('Date', 'FontSize', fs_axis, 'FontWeight', 'bold');
ylabel(sprintf('Realized %dY Excess Return', target_n), ...
    'FontSize', fs_axis, 'FontWeight', 'bold');

lgd = legend("True", "MODEL", "RW", "CS", "EH", 'Location', 'best');
lgd.FontSize = fs_legend;
lgd.Box = 'off';

%% =========================================================================
% Output struct for analysis
% =========================================================================

out = struct();

out.target_n = target_n;
out.dy_name = dy_name;

out.dy_dates = dates_dy;
out.dy_true  = dy_true;
out.dy_hat   = dy_hat;
out.dy_rw    = dy_rw;
out.dy_cum_sse_diff = cum_sse_diff;
out.dy_r2oos = r2oos_dy;
out.dy_pval  = pval_dy;

out.rx_dates = dates_rx;
out.rx_true  = rx_true;
out.rx_model = rx_model;
out.rx_rw    = rx_rw;
out.rx_cs    = rx_cs;
out.rx_eh    = rx_eh;

out.rx_r2_model = r2_model;
out.rx_r2_rw    = r2_rw;
out.rx_r2_cs    = r2_cs;
out.rx_r2_eh    = r2_eh;

out.rx_p_model = pvals(1);
out.rx_p_rw    = pvals(2);
out.rx_p_cs    = pvals(3);