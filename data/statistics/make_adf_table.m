function make_adf_table(in_mat, out_tex)
    % GENERATE_ADF_LATEX Performs Augmented Dickey-Fuller (ADF) tests on
    % Forward Rates and YoY changes in Forward Rates, and exports a
    % beautiful, publication-quality LaTeX table.
    %
    % Inputs:
    %   in_mat  - Path to target_and_features.mat (default: 'target_and_features.mat')
    %   out_tex - Path to save the LaTeX table (default: 'adf_table.tex')
    %
    % The script automatically selects the optimal lag length for each column
    % (maturities 2Y to 10Y) using the Akaike Information Criterion (AIC).
    % It uses native MATLAB adftest if available, and falls back to a custom
    % OLS ADF implementation if the Econometrics Toolbox is not installed.

    if nargin < 1 || isempty(in_mat)
        in_mat = 'target_and_features.mat';
    end
    if nargin < 2 || isempty(out_tex)
        out_tex = 'adf_table.tex';
    end

    % 1. Check file existence
    if ~exist(in_mat, 'file')
        parent_mat = fullfile('..', in_mat);
        if exist(parent_mat, 'file')
            in_mat = parent_mat;
        else
            error('File "%s" not found. Please run build_target_and_features first.', in_mat);
        end
    end

    fprintf('Loading data from "%s"...\n', in_mat);
    data_struct = load(in_mat);
    
    if ~isfield(data_struct, 'X')
        error('The file "%s" is missing expected structure X.', in_mat);
    end
    
    X = data_struct.X;
    fwd_data = X.fwd.data;
    yoy_fwd_data = X.yoy_fwd.data;

    % Restrict to the paper sample: August 1971 onward (YYYYMM >= 197108).
    sample_start = 197108;
    keep = X.fwd.Time >= sample_start;
    fwd_data = fwd_data(keep, :);
    yoy_fwd_data = yoy_fwd_data(keep, :);

    % Check toolbox
    has_toolbox = exist('adftest', 'file') == 2 || exist('adftest', 'file') == 3 || exist('adftest', 'file') == 5;
    if has_toolbox
        fprintf('Using native MATLAB Econometrics Toolbox for ADF tests.\n');
    else
        fprintf('Econometrics Toolbox not detected. Using custom OLS ADF engine.\n');
    end
    
    fprintf('Performing ADF tests on Forward Rates...\n');
    fwd_results = run_adf_block(fwd_data, has_toolbox);
    
    fprintf('Performing ADF tests on YoY changes in Forward Rates...\n');
    yoy_fwd_results = run_adf_block(yoy_fwd_data, has_toolbox);
    
    % Generate LaTeX code
    latex_code = build_latex_adf_table(fwd_results, yoy_fwd_results);
    
    % Print to Command Window
    fprintf('\n================== LaTeX ADF Table Code ==================\n');
    disp(latex_code);
    fprintf('==========================================================\n');
    
    % Write to .tex file
    fid = fopen(out_tex, 'w', 'n', 'UTF-8');
    if fid == -1
        warning('Could not open file "%s" for writing.', out_tex);
    else
        fprintf(fid, '%s', latex_code);
        fclose(fid);
        fprintf('\nLaTeX ADF table successfully written to "%s"\n', out_tex);
    end
end

function results = run_adf_block(data, has_toolbox)
    [~, N] = size(data);
    results = cell(1, N);
    
    for i = 1:N
        col = data(:, i);
        valid = col(~isnan(col));
        if length(valid) < 20
            results{i} = struct('stat', NaN, 'pval', NaN, 'lag', 0, 'nobs', 0);
            continue;
        end
        T_len = length(valid);
        max_lag = min(20, floor(12 * (T_len / 100)^0.25));
        
        best_aic = Inf;
        best_res = [];
        for lag = 0:max_lag
            if has_toolbox
                try
                    [~, pVal, stat, ~, reg] = adftest(valid, 'Model', 'ARD', 'Lags', lag);
                    aic = reg.numObs * log(reg.SSE / reg.numObs) + 2 * reg.numParams;
                    if aic < best_aic
                        best_aic = aic;
                        best_res = struct('stat', stat, 'pval', pVal, 'lag', lag, 'nobs', reg.numObs);
                    end
                catch
                    [stat, pVal, aic, nobs] = custom_adf_ols(valid, lag);
                    if aic < best_aic
                        best_aic = aic;
                        best_res = struct('stat', stat, 'pval', pVal, 'lag', lag, 'nobs', nobs);
                    end
                end
            else
                [stat, pVal, aic, nobs] = custom_adf_ols(valid, lag);
                if aic < best_aic
                    best_aic = aic;
                    best_res = struct('stat', stat, 'pval', pVal, 'lag', lag, 'nobs', nobs);
                end
            end
        end
        results{i} = best_res;
    end
end

function [stat, pval, aic, nobs] = custom_adf_ols(y, lag)
    dy = diff(y);
    T = length(y);
    Y = dy((lag+1):end);
    nobs = length(Y);
    X_const = ones(nobs, 1);
    X_level = y((lag+1):(T-1));
    X_diff = zeros(nobs, lag);
    for j = 1:lag
        X_diff(:, j) = dy((lag+1-j):(end-j));
    end
    Z = [X_const, X_level, X_diff];
    B = Z \ Y;
    residuals = Y - Z * B;
    SSE = residuals' * residuals;
    K = size(Z, 2);
    sigma2 = SSE / (nobs - K);
    V = sigma2 * inv(Z' * Z);
    stat = B(2) / sqrt(V(2, 2));
    aic = nobs * log(SSE / nobs) + 2 * K;
    pval = mackinnon_pval_drift(stat);
end

function pval = mackinnon_pval_drift(stat)
    mu = -1.8687;
    if stat < -5.0
        pval = 0.0;
        return;
    end
    if stat <= mu
        beta = [0.1039, 0.1587, 0.0543];
        z = stat;
        pval = 0.1039 + 0.1587 * z + 0.0543 * z^2;
        pval = normcdf(pval);
    else
        beta = [0.8252, 0.3541, 0.0526];
        z = stat;
        pval = 0.8252 + 0.3541 * z + 0.0526 * z^2;
        pval = normcdf(pval);
    end
    pval = max(0, min(1, pval));
    if stat > 1.0
        pval = 1.0;
    end
end

function latex = build_latex_adf_table(fwd_results, yoy_fwd_results)
    latex = sprintf('%% Augmented Dickey-Fuller (ADF) Test Table\n');
    latex = [latex, sprintf('\\begin{table}[tbp]\n')];
    latex = [latex, sprintf('  \\centering\n')];
    latex = [latex, sprintf('  \\caption{Augmented Dickey-Fuller (ADF) Unit Root Test Results}\n')];
    latex = [latex, sprintf('  \\label{tab:adf_test_results}\n');];
    latex = [latex, sprintf('  \\begin{tabular}{lccccccccc}\n')];
    latex = [latex, sprintf('    \\toprule\n')];
    latex = [latex, sprintf('    Maturity & 2Y & 3Y & 4Y & 5Y & 6Y & 7Y & 8Y & 9Y & 10Y \\\\\n')];
    latex = [latex, sprintf('    \\midrule\n')];
    
    % Panel A
    latex = [latex, sprintf('    \\multicolumn{10}{l}{\\textbf{Panel A: Forward Rates ($f_t^{(n)}$)}} \\\\\n')];
    latex = [latex, sprintf('    \\noalign{\\vskip 2pt}\n')];
    latex = [latex, format_adf_row('    ADF Stat', fwd_results, 'stat')];
    latex = [latex, format_adf_row('    p-value', fwd_results, 'pval')];
    latex = [latex, format_adf_row('    Lags (AIC)', fwd_results, 'lag')];
    latex = [latex, format_adf_row('    Observations', fwd_results, 'nobs')];
    latex = [latex, sprintf('    \\midrule\n')];
    
    % Panel B
    latex = [latex, sprintf('    \\multicolumn{10}{l}{\\textbf{Panel B: YoY Change in Forward Rates ($\\Delta f_t^{(n)}$)}} \\\\\n')];
    latex = [latex, sprintf('    \\noalign{\\vskip 2pt}\n')];
    latex = [latex, format_adf_row('    ADF Stat', yoy_fwd_results, 'stat')];
    latex = [latex, format_adf_row('    p-value', yoy_fwd_results, 'pval')];
    latex = [latex, format_adf_row('    Lags (AIC)', yoy_fwd_results, 'lag')];
    latex = [latex, format_adf_row('    Observations', yoy_fwd_results, 'nobs')];
    
    latex = [latex, sprintf('    \\bottomrule\n')];
    latex = [latex, sprintf('  \\end{tabular}\n')];
    latex = [latex, sprintf('  \\begin{tablenotes}[flushleft]\n')];
    latex = [latex, sprintf('    \\small\n')];
    latex = [latex, sprintf('    \\item \\textit{Note:} This table reports the Augmented Dickey-Fuller (ADF) unit root test statistics and corresponding $p$-values for the yield curve variables.\n')];
    latex = [latex, sprintf('    Panel A displays results for annual forward rates $f_t^{(n)}$, and Panel B displays results for the trailing 12-month change in forward rates $\\Delta f_t^{(n)} = f_t^{(n)} - f_{t-12}^{(n)}$.\n')];
    latex = [latex, sprintf('    All models include an intercept (drift term). Lag lengths are selected automatically for each series using the Akaike Information Criterion (AIC).\n')];
    latex = [latex, sprintf('    Asymptotic critical values for the constant-only model are $-$3.4388 (1\\%% significance level), $-$2.8653 (5\\%% significance level), and $-$2.5688 (10\\%% significance level).\n')];
    latex = [latex, sprintf('  \\end{tablenotes}\n')];
    latex = [latex, sprintf('\\end{table}\n')];
end

function str = format_adf_row(label, results, field)
    str = sprintf('%%-18s', label);
    for i = 1:length(results)
        v = results{i}.(field);
        if strcmp(field, 'lag') || strcmp(field, 'nobs')
            if isnan(v)
                str = [str, ' &   NaN '];
            else
                str = [str, sprintf(' & %%8d', int32(v))];
            end
        else
            if isnan(v)
                str = [str, ' &   NaN '];
            else
                str = [str, sprintf(' & %%8.4f', v)];
            end
        end
    end
    str = [str, sprintf(' \\\\\n')];
end
