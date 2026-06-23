function make_summary_table(in_mat, out_tex)
    % GENERATE_SUMMARY_LATEX Loads yield data, computes summary statistics,
    % and exports a beautiful, publication-quality LaTeX table (booktabs style).
    %
    % Inputs:
    %   in_mat  - Path to target_and_features.mat (default: 'target_and_features.mat')
    %   out_tex - Path to save the LaTeX table (default: 'summary_table.tex')
    %
    % The output table includes three panels:
    %   Panel A: One-Year Excess Returns rx^(n) (in percent)
    %   Panel B: Forward Rates f^(n) (converted to percent)
    %   Panel C: YoY Change in Forward Rates \Delta f^(n) (converted to percent)
    %
    % Statistics computed:
    %   - Mean (평균)
    %   - Standard Deviation (표준편차)
    %   - Minimum (최솟값)
    %   - Maximum (최댓값)
    %   - AR(1) Autocorrelation (시차 1 자기상관계수)
    %   - Skewness (왜도)
    %   - Kurtosis (첨도)

    if nargin < 1 || isempty(in_mat)
        in_mat = 'target_and_features.mat';
    end
    if nargin < 2 || isempty(out_tex)
        out_tex = 'summary_table.tex';
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
    
    if ~isfield(data_struct, 'X') || ~isfield(data_struct, 'y')
        error('The file "%s" is missing expected structures X or y.', in_mat);
    end
    
    X = data_struct.X;
    y = data_struct.y;
    
    rx_data = y.rx.data;
    fwd_data = X.fwd.data * 100;
    yoy_fwd_data = X.yoy_fwd.data * 100;
    dy_data = y.dy.data;   % Carry-Anchored target dy_1..dy_9, already in percent

    % Restrict to the paper sample: August 1971 onward (YYYYMM >= 197108).
    sample_start = 197108;
    keep = X.fwd.Time >= sample_start;
    rx_data = rx_data(keep, :);
    fwd_data = fwd_data(keep, :);
    yoy_fwd_data = yoy_fwd_data(keep, :);
    dy_data = dy_data(keep, :);

    % Calculate statistics
    rx_stats = calculate_block_stats(rx_data);
    fwd_stats = calculate_block_stats(fwd_data);
    yoy_fwd_stats = calculate_block_stats(yoy_fwd_data);
    dy_stats = calculate_block_stats(dy_data);

    % Generate LaTeX code
    latex_code = build_latex_table(rx_stats, fwd_stats, yoy_fwd_stats, dy_stats);
    
    % Print to Command Window
    fprintf('\n================== LaTeX Table Code ==================\n');
    disp(latex_code);
    fprintf('======================================================\n');
    
    % Write to .tex file
    fid = fopen(out_tex, 'w', 'n', 'UTF-8');
    if fid == -1
        warning('Could not open file "%s" for writing.', out_tex);
    else
        fprintf(fid, '%s', latex_code);
        fclose(fid);
        fprintf('\nLaTeX table successfully written to "%s"\n', out_tex);
    end
end

function stats = calculate_block_stats(data)
    [~, N] = size(data);
    stats.mean = zeros(1, N);
    stats.std = zeros(1, N);
    stats.min = zeros(1, N);
    stats.max = zeros(1, N);
    stats.ar1 = zeros(1, N);
    stats.skew = zeros(1, N);
    stats.kurt = zeros(1, N);
    
    for i = 1:N
        col = data(:, i);
        valid = col(~isnan(col));
        if isempty(valid)
            stats.mean(i) = NaN;
            stats.std(i) = NaN;
            stats.min(i) = NaN;
            stats.max(i) = NaN;
            stats.ar1(i) = NaN;
            stats.skew(i) = NaN;
            stats.kurt(i) = NaN;
            continue;
        end
        stats.mean(i) = mean(valid);
        stats.std(i) = std(valid);
        stats.min(i) = min(valid);
        stats.max(i) = max(valid);
        
        x_t = col(2:end);
        x_t1 = col(1:end-1);
        ok = ~isnan(x_t) & ~isnan(x_t1);
        if sum(ok) > 1
            R = corrcoef(x_t(ok), x_t1(ok));
            stats.ar1(i) = R(1, 2);
        else
            stats.ar1(i) = NaN;
        end
        [skewVal, kurtVal] = my_skew_kurt(valid);
        stats.skew(i) = skewVal;
        stats.kurt(i) = kurtVal;
    end
end

function [skewVal, kurtVal] = my_skew_kurt(x)
    N = length(x);
    if N < 3
        skewVal = NaN;
        kurtVal = NaN;
        return;
    end
    mu = mean(x);
    m2 = mean((x - mu).^2);
    m3 = mean((x - mu).^3);
    m4 = mean((x - mu).^4);
    skewVal = m3 / (m2^(1.5));
    kurtVal = m4 / (m2^2);
end

function latex = build_latex_table(rx_stats, fwd_stats, yoy_fwd_stats, dy_stats)
    latex = sprintf('%% Publication-quality Yield Curve Summary Statistics Table\n');
    latex = [latex, sprintf('\\begin{table}[tbp]\n')];
    latex = [latex, sprintf('  \\centering\n')];
    latex = [latex, sprintf('  \\caption{Summary Statistics for Yield Curve Excess Returns and Forwards}\n')];
    latex = [latex, sprintf('  \\label{tab:yield_summary_stats}\n')];
    latex = [latex, sprintf('  \\begin{tabular}{lccccccccc}\n')];
    latex = [latex, sprintf('    \\toprule\n')];
    latex = [latex, sprintf('    Maturity & 2Y & 3Y & 4Y & 5Y & 6Y & 7Y & 8Y & 9Y & 10Y \\\\\n')];
    latex = [latex, sprintf('    \\midrule\n')];
    
    % Panel A
    latex = [latex, sprintf('    \\multicolumn{10}{l}{\\textbf{Panel A: One-Year Excess Returns ($rx_t^{(n)}$, \\%%)}} \\\\\n')];
    latex = [latex, sprintf('    \\noalign{\\vskip 2pt}\n')];
    latex = [latex, format_row('    Mean', rx_stats.mean)];
    latex = [latex, format_row('    Std. Dev.', rx_stats.std)];
    latex = [latex, format_row('    Min', rx_stats.min)];
    latex = [latex, format_row('    Max', rx_stats.max)];
    latex = [latex, format_row('    AR(1)', rx_stats.ar1)];
    latex = [latex, format_row('    Skewness', rx_stats.skew)];
    latex = [latex, format_row('    Kurtosis', rx_stats.kurt)];
    latex = [latex, sprintf('    \\midrule\n')];
    
    % Panel B
    latex = [latex, sprintf('    \\multicolumn{10}{l}{\\textbf{Panel B: Forward Rates ($f_t^{(n)}$, \\%%)}} \\\\\n')];
    latex = [latex, sprintf('    \\noalign{\\vskip 2pt}\n')];
    latex = [latex, format_row('    Mean', fwd_stats.mean)];
    latex = [latex, format_row('    Std. Dev.', fwd_stats.std)];
    latex = [latex, format_row('    Min', fwd_stats.min)];
    latex = [latex, format_row('    Max', fwd_stats.max)];
    latex = [latex, format_row('    AR(1)', fwd_stats.ar1)];
    latex = [latex, format_row('    Skewness', fwd_stats.skew)];
    latex = [latex, format_row('    Kurtosis', fwd_stats.kurt)];
    latex = [latex, sprintf('    \\midrule\n')];
    
    % Panel C
    latex = [latex, sprintf('    \\multicolumn{10}{l}{\\textbf{Panel C: YoY Change in Forward Rates ($\\Delta f_t^{(n)}$, \\%%)}} \\\\\n')];
    latex = [latex, sprintf('    \\noalign{\\vskip 2pt}\n')];
    latex = [latex, format_row('    Mean', yoy_fwd_stats.mean)];
    latex = [latex, format_row('    Std. Dev.', yoy_fwd_stats.std)];
    latex = [latex, format_row('    Min', yoy_fwd_stats.min)];
    latex = [latex, format_row('    Max', yoy_fwd_stats.max)];
    latex = [latex, format_row('    AR(1)', yoy_fwd_stats.ar1)];
    latex = [latex, format_row('    Skewness', yoy_fwd_stats.skew)];
    latex = [latex, format_row('    Kurtosis', yoy_fwd_stats.kurt)];
    latex = [latex, sprintf('    \\midrule\n')];

    % Panel D
    latex = [latex, sprintf('    \\multicolumn{10}{l}{\\textbf{Panel D: Rolled-Maturity Yield-Change Target ($\\Delta y_{t+1}^{(n-1)}$, \\%%)}} \\\\\n')];
    latex = [latex, sprintf('    \\noalign{\\vskip 2pt}\n')];
    latex = [latex, format_row('    Mean', dy_stats.mean)];
    latex = [latex, format_row('    Std. Dev.', dy_stats.std)];
    latex = [latex, format_row('    Min', dy_stats.min)];
    latex = [latex, format_row('    Max', dy_stats.max)];
    latex = [latex, format_row('    AR(1)', dy_stats.ar1)];
    latex = [latex, format_row('    Skewness', dy_stats.skew)];
    latex = [latex, format_row('    Kurtosis', dy_stats.kurt)];

    latex = [latex, sprintf('    \\bottomrule\n')];
    latex = [latex, sprintf('  \\end{tabular}\n')];
    latex = [latex, sprintf('  \\begin{tablenotes}[flushleft]\n')];
    latex = [latex, sprintf('    \\small\n')];
    latex = [latex, sprintf('    \\item \\textit{Note:} This table presents the summary statistics for the yield curve variables computed from the monthly dataset.\n')];
    latex = [latex, sprintf('    Panel A displays the one-year excess returns ($rx_t^{(n)} = n y_t^{(n)} - (n-1) y_{t+12}^{(n-1)} - y_t^{(1)}$) for annual maturities $n = 2, \\dots, 10$.\n')];
    latex = [latex, sprintf('    Panel B displays the annual forward rates ($f_t^{(n)} = n y_t^{(n)} - (n-1) y_t^{(n-1)}$).\n')];
    latex = [latex, sprintf('    Panel C displays the trailing 12-month change in forward rates ($\\Delta f_t^{(n)} = f_t^{(n)} - f_{t-12}^{(n)}$).\n')];
    latex = [latex, sprintf('    Panel D displays the rolled-maturity yield-change target ($\\Delta y_{t+1}^{(n-1)} = y_{t+1}^{(n-1)} - y_t^{(n-1)}$), reported under each bond maturity $n$.\n')];
    latex = [latex, sprintf('    All variables are expressed in percentage terms (\\%%). AR(1) represents the first-order autocorrelation coefficient.\n')];
    latex = [latex, sprintf('  \\end{tablenotes}\n')];
    latex = [latex, sprintf('\\end{table}\n')];
end

function str = format_row(label, row_data)
    str = sprintf('%%-18s', label);
    for v = row_data
        if isnan(v)
            str = [str, ' &   NaN '];
        else
            str = [str, sprintf(' & %%8.4f', v)];
        end
    end
    str = [str, sprintf(' \\\\\n')];
end
