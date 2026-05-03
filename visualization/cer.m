clear; clc; close all;

%% ============================================================
% User settings
%% ============================================================
res_file   = 'ridge_pc2.mat';          % ycNN / ycGLM / ycOLS result file
dataset_file = 'dataset.csv';          % raw yield csv (percent units)
rv_file      = 'rv_LW_monthly.xlsx';   % RV file (decimal units)

target_n = 10;                         % e.g. 2,5,7,10
gamma_list = [2, 10];

first_forecast_origin = datetime(1989,1,31);
last_forecast_origin  = datetime(2023,12,31);

h = 12;                                % 12-month horizon
weight_bounds = [];                    % [] = unconstrained

% Optional:
% Leave empty to auto-detect the unique field starting with Y_forecast_agg_
model_field_override = "";

%% ============================================================
% Step 0. Load result file and detect forecast field
%% ============================================================
S = load(res_file);

assert(isfield(S, 'Dates'),     'Dates is missing in result file.');
assert(isfield(S, 'Y_Columns'), 'Y_Columns is missing in result file.');
assert(isfield(S, 'Y_True'),    'Y_True is missing in result file.');

if strlength(model_field_override) > 0
    agg_field = char(model_field_override);
    assert(isfield(S, agg_field), 'Missing field %s in result file.', agg_field);
else
    fn = string(fieldnames(S));
    agg_candidates = fn(startsWith(fn, "Y_forecast_agg_"));

    if isempty(agg_candidates)
        error('No field starting with Y_forecast_agg_ found in result file.');
    elseif numel(agg_candidates) > 1
        disp('Available forecast fields:');
        disp(agg_candidates);
        error('Multiple Y_forecast_agg_ fields found. Set model_field_override explicitly.');
    else
        agg_field = char(agg_candidates(1));
    end
end

model_name = string(erase(string(agg_field), "Y_forecast_agg_"));
fprintf('Using forecast field: %s\n', agg_field);
fprintf('Detected model name : %s\n', model_name);

%% ============================================================
% Step 1. Parse result-file dates, columns, forecasts
%% ============================================================
if isdatetime(S.Dates)
    dates_res = S.Dates(:);
elseif isstring(S.Dates)
    dates_res = datetime(S.Dates(:));
elseif iscell(S.Dates)
    dates_res = datetime(string(S.Dates(:)));
elseif ischar(S.Dates)
    dates_res = datetime(string(cellstr(S.Dates)));
else
    error('Unsupported Dates format in result file.');
end

if isstring(S.Y_Columns)
    ycols = S.Y_Columns(:);
elseif iscell(S.Y_Columns)
    ycols = string(S.Y_Columns(:));
elseif ischar(S.Y_Columns)
    ycols = string(cellstr(S.Y_Columns));
else
    error('Unsupported Y_Columns format in result file.');
end

Y_true = S.Y_True;
Y_hat  = S.(agg_field);

assert(size(Y_true,1) == numel(dates_res), 'Y_True and Dates are inconsistent.');
assert(size(Y_hat,1)  == numel(dates_res), 'Y_hat and Dates are inconsistent.');

dy_idx  = target_n - 1;
dy_name = "dy_" + string(dy_idx);

j_dy = find(ycols == dy_name, 1);
if isempty(j_dy)
    error('Missing %s in Y_Columns.', dy_name);
end

dy_hat_full = Y_hat(:, j_dy);   % percent units
dy_true_full = Y_true(:, j_dy); %#ok<NASGU>

%% ============================================================
% Step 2. Load monthly yields from dataset.csv
%% ============================================================
D = readtable(dataset_file);
D.Date = eom_from_yyyymm(D.Time);

col_y1   = maturity_col(12);
col_ynm1 = maturity_col(12 * (target_n - 1));
col_yn   = maturity_col(12 * target_n);

assert(ismember(col_y1,   D.Properties.VariableNames), 'Missing %s.', col_y1);
assert(ismember(col_ynm1, D.Properties.VariableNames), 'Missing %s.', col_ynm1);
assert(ismember(col_yn,   D.Properties.VariableNames), 'Missing %s.', col_yn);

% Keep yields in percent units here for clean consistency with dy_hat_full.
D.y1_pct   = D.(col_y1);
D.ynm1_pct = D.(col_ynm1);
D.yn_pct   = D.(col_yn);

%% ============================================================
% Step 3. Construct realized rx, RW forecast, EH forecast
%% ============================================================
% Realized target with origin t:
%   rx_t^(n) = n*y_t^(n) - (n-1)*y_{t+12}^{(n-1)} - y_t^(1)
D.ynm1_lead_pct = [D.ynm1_pct((1+h):end); nan(h,1)];
D.rx_realized_pct = target_n * D.yn_pct ...
                  - (target_n - 1) * D.ynm1_lead_pct ...
                  - D.y1_pct;

% RW forecast at origin t:
%   E_t[y_{t+12}^{(n-1)}] = y_t^{(n-1)}
D.mu_rw_pct = target_n * D.yn_pct ...
            - (target_n - 1) * D.ynm1_pct ...
            - D.y1_pct;

% EH forecast with strict embargo:
% use realized returns with origin <= t-12
D.mu_eh_pct = nan(height(D), 1);

for j = 1:height(D)
    cutoff_date = D.Date(j) - calmonths(h);
    use_idx = (D.Date <= cutoff_date) & isfinite(D.rx_realized_pct);
    D.mu_eh_pct(j) = mean(D.rx_realized_pct(use_idx), 'omitnan');
end

%% ============================================================
% Step 4. Align model dy forecast with raw yield panel
%% ============================================================
[dates_common, ia, ib] = intersect(dates_res, D.Date, 'stable');

dy_hat_pct = dy_hat_full(ia);
D_common = D(ib, :);
D_common.Date = dates_common;

% Model-implied forecast for y_{t+12}^{(n-1)} in percent:
%   y_{t+12|t}^{(n-1)} = y_t^{(n-1)} + \widehat{dy}_{t}^{(n-1)}
D_common.ynm1_model_lead_pct = D_common.ynm1_pct + dy_hat_pct;

% Model-implied rx forecast in percent:
D_common.mu_model_pct = target_n * D_common.yn_pct ...
                      - (target_n - 1) * D_common.ynm1_model_lead_pct ...
                      - D_common.y1_pct;

%% ============================================================
% Step 5. Build common forecast-origin sample
%% ============================================================
valid_origin = ...
    D_common.Date >= first_forecast_origin & ...
    D_common.Date <= last_forecast_origin  & ...
    isfinite(D_common.rx_realized_pct)     & ...
    isfinite(D_common.mu_eh_pct)           & ...
    isfinite(D_common.mu_rw_pct)           & ...
    isfinite(D_common.mu_model_pct);

F = D_common(valid_origin, { ...
    'Date','Time', ...
    'mu_model_pct','mu_rw_pct','mu_eh_pct','rx_realized_pct'});

%% ============================================================
% Step 6. Forecast comparison on common sample
%% ============================================================
F.e_model = F.rx_realized_pct - F.mu_model_pct;
F.e_rw    = F.rx_realized_pct - F.mu_rw_pct;
F.e_eh    = F.rx_realized_pct - F.mu_eh_pct;

F.sse_model = F.e_model .^ 2;
F.sse_rw    = F.e_rw    .^ 2;
F.sse_eh    = F.e_eh    .^ 2;

R2_model_vs_eh = 1 - sum(F.sse_model, 'omitnan') / sum(F.sse_eh, 'omitnan');
R2_rw_vs_eh    = 1 - sum(F.sse_rw,    'omitnan') / sum(F.sse_eh, 'omitnan');

F.delta_sse_model_eh = F.sse_eh - F.sse_model;
F.delta_sse_rw_eh    = F.sse_eh - F.sse_rw;

F.cum_delta_sse_model_eh = cumsum(F.delta_sse_model_eh);
F.cum_delta_sse_rw_eh    = cumsum(F.delta_sse_rw_eh);

rx_bar_pct = mean(F.rx_realized_pct, 'omitnan');
sse_denom  = sum((F.rx_realized_pct - rx_bar_pct).^2, 'omitnan');

F.cum_delta_sse_model_eh_norm = F.cum_delta_sse_model_eh / sse_denom;
F.cum_delta_sse_rw_eh_norm    = F.cum_delta_sse_rw_eh    / sse_denom;

%% ============================================================
% Step 7. Load RV and build sigma^2
%% ============================================================
RV = readtable(rv_file);

if ~isdatetime(RV.Date)
    RV.Date = datetime(string(RV.Date), 'InputFormat', 'yyyy-MM-dd');
end

rv_col = maturity_col(12 * (target_n - 1));
assert(ismember(rv_col, RV.Properties.VariableNames), 'Missing %s in RV file.', rv_col);

RV.rv_nminus1 = RV.(rv_col);                  % decimal units
RV.sigma2     = (target_n - 1)^2 * (RV.rv_nminus1 .^ 2);

RV = RV(:, {'Date','Time','rv_nminus1','sigma2'});

%% ============================================================
% Step 8. Merge forecast sample with RV for CER
%% ============================================================
Tcer = innerjoin( ...
    F(:, {'Date','Time','mu_model_pct','mu_rw_pct','mu_eh_pct','rx_realized_pct'}), ...
    RV, ...
    'Keys', {'Date','Time'} ...
);

valid_cer = ...
    isfinite(Tcer.mu_model_pct)    & ...
    isfinite(Tcer.mu_rw_pct)       & ...
    isfinite(Tcer.mu_eh_pct)       & ...
    isfinite(Tcer.rx_realized_pct) & ...
    isfinite(Tcer.sigma2);

Tcer = Tcer(valid_cer, :);

% Convert rx forecasts/realizations from percent to decimals for CER
Tcer.mu_model = Tcer.mu_model_pct / 100;
Tcer.mu_rw    = Tcer.mu_rw_pct    / 100;
Tcer.mu_eh    = Tcer.mu_eh_pct    / 100;
Tcer.rx_realized = Tcer.rx_realized_pct / 100;

%% ============================================================
% Step 9. Compute CER paths for each gamma
%% ============================================================
results = struct();

for g = 1:numel(gamma_list)
    gamma = gamma_list(g);
    G = Tcer;

    % Optimal weights
    G.w_model = (G.mu_model + 0.5 * G.sigma2) ./ (gamma * G.sigma2);
    G.w_rw    = (G.mu_rw    + 0.5 * G.sigma2) ./ (gamma * G.sigma2);
    G.w_eh    = (G.mu_eh    + 0.5 * G.sigma2) ./ (gamma * G.sigma2);

    if ~isempty(weight_bounds)
        G.w_model = min(max(G.w_model, weight_bounds(1)), weight_bounds(2));
        G.w_rw    = min(max(G.w_rw,    weight_bounds(1)), weight_bounds(2));
        G.w_eh    = min(max(G.w_eh,    weight_bounds(1)), weight_bounds(2));
    end

    % Period-by-period realized CER
    G.u_model = (G.rx_realized + 0.5 * G.sigma2) .* G.w_model ...
              - 0.5 * gamma * G.sigma2 .* (G.w_model .^ 2);

    G.u_rw    = (G.rx_realized + 0.5 * G.sigma2) .* G.w_rw ...
              - 0.5 * gamma * G.sigma2 .* (G.w_rw .^ 2);

    G.u_eh    = (G.rx_realized + 0.5 * G.sigma2) .* G.w_eh ...
              - 0.5 * gamma * G.sigma2 .* (G.w_eh .^ 2);

    % Utility gains
    G.du_model_rw = G.u_model - G.u_rw;
    G.du_model_eh = G.u_model - G.u_eh;
    G.du_rw_eh    = G.u_rw    - G.u_eh;

    % Cumulative paths
    G.cu_model = cumsum(G.u_model);
    G.cu_rw    = cumsum(G.u_rw);
    G.cu_eh    = cumsum(G.u_eh);

    G.cu_gain_model_rw = cumsum(G.du_model_rw);
    G.cu_gain_model_eh = cumsum(G.du_model_eh);
    G.cu_gain_rw_eh    = cumsum(G.du_rw_eh);

    fld = sprintf('gamma_%d', gamma);
    results.(fld) = G;
end

%% ============================================================
% Step 10. Summary table
%% ============================================================
summary_tbl = table();
summary_tbl.gamma = gamma_list(:);

avg_u_model = nan(numel(gamma_list),1);
avg_u_rw    = nan(numel(gamma_list),1);
avg_u_eh    = nan(numel(gamma_list),1);

cum_u_model = nan(numel(gamma_list),1);
cum_u_rw    = nan(numel(gamma_list),1);
cum_u_eh    = nan(numel(gamma_list),1);

for g = 1:numel(gamma_list)
    fld = sprintf('gamma_%d', gamma_list(g));
    G = results.(fld);

    avg_u_model(g) = mean(G.u_model, 'omitnan');
    avg_u_rw(g)    = mean(G.u_rw,    'omitnan');
    avg_u_eh(g)    = mean(G.u_eh,    'omitnan');

    cum_u_model(g) = G.cu_model(end);
    cum_u_rw(g)    = G.cu_rw(end);
    cum_u_eh(g)    = G.cu_eh(end);
end

summary_tbl.avg_u_model = avg_u_model;
summary_tbl.avg_u_rw    = avg_u_rw;
summary_tbl.avg_u_eh    = avg_u_eh;
summary_tbl.cum_u_model = cum_u_model;
summary_tbl.cum_u_rw    = cum_u_rw;
summary_tbl.cum_u_eh    = cum_u_eh;

disp('Summary table:');
disp(summary_tbl);

fprintf('\nForecast R2 versus EH benchmark:\n');
fprintf('MODEL vs EH: %.6f\n', R2_model_vs_eh);
fprintf('RW    vs EH: %.6f\n', R2_rw_vs_eh);

%% ============================================================
% Step 11. Figures
%% ============================================================
assert(isfield(results, 'gamma_2'),  'results.gamma_2 is missing.');
assert(isfield(results, 'gamma_10'), 'results.gamma_10 is missing.');

G2  = results.gamma_2;
G10 = results.gamma_10;

% CER across time: gamma = 2
figure('Color','w','Name',sprintf('CER across time, gamma=%d', 2));
plot(G2.Date, G2.u_model, 'LineWidth', 1.8); hold on;
plot(G2.Date, G2.u_rw,    'LineWidth', 1.8);
plot(G2.Date, G2.u_eh,    'LineWidth', 1.5);
yline(0, 'k-');
grid on;
legend({sprintf('CER^{%s}', model_name), 'CER^{RW}', 'CER^{EH}'}, 'Location', 'best');
xlabel('Date', 'FontSize', 16);
ylabel('CER', 'FontSize', 16);
set(gca, 'FontSize', 14);

% CER across time: gamma = 10
figure('Color','w','Name',sprintf('CER across time, gamma=%d', 10));
plot(G10.Date, G10.u_model, 'LineWidth', 1.8); hold on;
plot(G10.Date, G10.u_rw,    'LineWidth', 1.8);
plot(G10.Date, G10.u_eh,    'LineWidth', 1.5);
yline(0, 'k-');
grid on;
legend({sprintf('CER^{%s}', model_name), 'CER^{RW}', 'CER^{EH}'}, 'Location', 'best');
xlabel('Date', 'FontSize', 16);
ylabel('CER', 'FontSize', 16);
set(gca, 'FontSize', 14);

% Cumulative CER: gamma = 2 and 10
figure('Color','w','Name',sprintf('Cumulative CER, n=%d', target_n));
plot(G2.Date,  G2.cu_model, '-',  'LineWidth', 1.8); hold on;
plot(G2.Date,  G2.cu_rw,    '-',  'LineWidth', 1.8);
plot(G2.Date,  G2.cu_eh,    '--', 'LineWidth', 1.5);
plot(G10.Date, G10.cu_model,'-',  'LineWidth', 1.8);
plot(G10.Date, G10.cu_rw,   '-',  'LineWidth', 1.8);
plot(G10.Date, G10.cu_eh,   '--', 'LineWidth', 1.5);
yline(0, 'k-');
grid on;
legend( ...
    {sprintf('%s, \\gamma=2', model_name), 'RW, \gamma=2', 'EH, \gamma=2', ...
     sprintf('%s, \\gamma=10', model_name), 'RW, \gamma=10', 'EH, \gamma=10'}, ...
    'Location', 'best');
xlabel('Date', 'FontSize', 16);
ylabel('Cumulative CER', 'FontSize', 16);
set(gca, 'FontSize', 14);

% Normalized cumulative SSE difference vs EH
figure('Color','w','Name','Normalized cumulative SSE difference vs EH');
plot(F.Date, F.cum_delta_sse_model_eh_norm, 'LineWidth', 1.8); hold on;
plot(F.Date, F.cum_delta_sse_rw_eh_norm,    'LineWidth', 1.8);
yline(0, 'k-');
grid on;
legend({sprintf('%s - EH', model_name), 'RW - EH'}, 'Location', 'best');
xlabel('Date', 'FontSize', 16);
ylabel('Normalized cumulative SSE difference', 'FontSize', 16);
set(gca, 'FontSize', 14);

%% ============================================================
% Step 12. Save output
%% ============================================================
out_file = sprintf('cer_%s_vs_rw_eh_n%d.mat', char(model_name), target_n);
save(out_file, ...
    'F', 'Tcer', 'results', 'summary_tbl', ...
    'target_n', 'gamma_list', ...
    'first_forecast_origin', 'last_forecast_origin', ...
    'R2_model_vs_eh', 'R2_rw_vs_eh', ...
    'weight_bounds', 'res_file', 'agg_field');

fprintf('Saved output: %s\n', out_file);

%% ============================================================
% Local functions
%% ============================================================
function dt = eom_from_yyyymm(yyyymm)
    y = floor(double(yyyymm) / 100);
    m = mod(double(yyyymm), 100);
    dt = datetime(y, m, 1);
    dt = dateshift(dt, 'end', 'month');
end

function name = maturity_col(months)
    name = sprintf('m%03d', months);
end