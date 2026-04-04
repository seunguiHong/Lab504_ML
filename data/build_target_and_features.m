% =========================================================================
% Date      : 2026-03-12
% Input     : dataset.csv
% Output    : target_and_features.mat
% Objective : Save grouped predictors X.* (decimals) and targets y.* (percent)
%             in a Python-friendly -v7 MAT format.
%
% Conventions
%   - Yield columns m001..m360 are assumed to be in percent units in CSV.
%   - X.* is stored in decimals.
%   - y.* is stored in percent.
%   - Annual nodes are m012..m120 (1y..10y) in decimals.
%   - X.fwd includes fwd_1 = m012 (1y zero rate) in decimals.
%   - X.slopechg stores annual changes in slope:
%       ds_n(t) = s_n(t) - s_n(t-12),  n = 2..10
%       where s_n(t) = y_t^(n) - y_t^(1)
%   - X.dy_pc stores PC1-PC3 extracted from dy over m001..m120, where
%       dy_m(t) = y_t^(m) - y_{t-12}^(m),  m = 1,...,120
%     and the PCA at time t is estimated using only rows available up to t.
%   - X.dy_pc1, X.dy_pc2, X.dy_pc3 store the individual PC scores as
%     separate one-column feature groups.
%   - Target y.dy is the 12m-ahead same constant-maturity yield change:
%       dy_n(t) = y_{t+12}^{(n)} - y_t^{(n)}, n = 1..10
%
% Save format
%   - Each group is a plain struct with fields:
%       .Time
%       .data
%       .names
%   - Saved with -v7 for direct scipy.io.loadmat compatibility.
% =========================================================================

function build_target_and_features(in_csv, out_mat)

    if nargin < 1 || isempty(in_csv),  in_csv = 'dataset.csv'; end
    if nargin < 2 || isempty(out_mat), out_mat = 'target_and_features.mat'; end

    % ---------------------------
    % 1) Load raw CSV
    % ---------------------------
    opts = detectImportOptions(in_csv, 'VariableNamingRule', 'preserve');
    raw  = readtable(in_csv, opts);

    vn = string(raw.Properties.VariableNames);
    if ~ismember("Time", vn)
        error("Missing required column: Time");
    end

    Time = raw.Time(:);

    % ---------------------------
    % 2) Yields: m001..m360
    % ---------------------------
    y_mask = ~cellfun('isempty', regexp(vn, '^m\d{3}$', 'once'));
    if sum(y_mask) < 120
        error("Insufficient yield columns: need at least m001..m120.");
    end

    T_Yields = raw(:, y_mask);
    Y_all    = table2array(T_Yields) / 100;   % percent -> decimal

    if size(Y_all, 2) < 120
        error("Yield matrix has fewer than 120 monthly maturities.");
    end

    % Annual nodes: 1y..10y from m012..m120 (decimals)
    y_annual = Y_all(:, 12:12:120);   % T x 10

    % Monthly nodes: m001..m120 for dy-PCA
    y_120 = Y_all(:, 1:120);          % T x 120

    % ---------------------------
    % 3) Derived features
    % ---------------------------
    % Slope: s_n = y(n) - y(1), n = 2..10
    S = y_annual(:, 2:end) - y_annual(:, 1);

    % Annual slope change: ds_n(t) = s_n(t) - s_n(t-12), n = 2..10
    S_chg = NaN(size(S));
    if size(S, 1) > 12
        S_chg(13:end, :) = S(13:end, :) - S(1:end-12, :);
    end

    % Forward: fwd_n = n*y(n) - (n-1)*y(n-1), n = 2..10
    n_vec = 2:10;
    FWD   = y_annual(:, 2:end) .* n_vec - y_annual(:, 1:end-1) .* (n_vec - 1);

    % 1y zero rate
    rf12 = y_annual(:, 1);

    % dy over m001..m120: dy_m(t) = y_t^(m) - y_{t-12}^(m)
    T = size(y_120, 1);
    DY_120 = NaN(T, 120);
    if T > 12
        DY_120(13:end, :) = y_120(13:end, :) - y_120(1:end-12, :);
    end

    % Expanding-window PCA scores from DY_120
    % At time t, PCA is estimated using only rows 13:t.
    % PCA is implemented via SVD to avoid unnecessary TSQUARED computation.
    DY_PC = NaN(T, 3);
    prev_V = [];

    % Minimum number of training rows before extracting PCs
    min_obs_pca = 24;

    for t = 13:T
        D_train = DY_120(13:t, :);
        ok_train = all(isfinite(D_train), 2);
        D_train = D_train(ok_train, :);

        if size(D_train, 1) < min_obs_pca
            continue;
        end

        x_t = DY_120(t, :);
        if ~all(isfinite(x_t))
            continue;
        end

        mu = mean(D_train, 1);
        D_center = D_train - mu;

        [~, ~, V] = svd(D_center, 'econ');

        n_keep = min(3, size(V, 2));
        if n_keep < 1
            continue;
        end

        V_use = V(:, 1:n_keep);

        % Sign alignment for time-series consistency
        if ~isempty(prev_V)
            n_align = min(size(prev_V, 2), size(V_use, 2));
            for k = 1:n_align
                if prev_V(:, k)' * V_use(:, k) < 0
                    V_use(:, k) = -V_use(:, k);
                end
            end
        end

        score_t = (x_t - mu) * V_use;
        DY_PC(t, 1:n_keep) = score_t;

        prev_V = V_use;
    end

    DY_PC1 = DY_PC(:, 1);
    DY_PC2 = DY_PC(:, 2);
    DY_PC3 = DY_PC(:, 3);

    % ---------------------------
    % 4) External columns
    % ---------------------------
    T_Ext  = raw(:, ~y_mask & vn ~= "Time");
    ext_vn = string(T_Ext.Properties.VariableNames);

    is_iv      = startsWith(ext_vn, "ATM_IV_");
    is_macropc = ~cellfun('isempty', regexp(ext_vn, '^F\d+(\^3)?$', 'once'));
    is_macro   = ~(is_iv | is_macropc);

    % ---------------------------
    % 5) Target: 12m-ahead same-maturity yield change
    % ---------------------------
    % Keep both dy_1 and dy_10, so targets span 1y..10y.
    DY_dec = NaN(T, 10);

    if T > 12
        DY_dec(1:T-12, :) = y_annual(13:T, 1:10) - y_annual(1:T-12, 1:10);
    end

    DY_pct = 100 * DY_dec;  % decimal -> percent

    % ---------------------------
    % 6) Assemble outputs
    % ---------------------------
    X = struct();
    y = struct();

    X.time = Time;
    y.time = Time;

    X.slope.Time  = Time;
    X.slope.data  = S;
    X.slope.names = cellstr(compose('s_%d', 2:10));

    X.slopechg.Time  = Time;
    X.slopechg.data  = S_chg;
    X.slopechg.names = cellstr(compose('ds_%d', 2:10));

    X.fwd.Time  = Time;
    X.fwd.data  = [rf12, FWD];
    X.fwd.names = cellstr(["fwd_1", compose('fwd_%d', 2:10)]);

    X.dy_pc.Time  = Time;
    X.dy_pc.data  = DY_PC;
    X.dy_pc.names = {'dy_pc_1', 'dy_pc_2', 'dy_pc_3'};

    X.dy_pc1.Time  = Time;
    X.dy_pc1.data  = DY_PC1;
    X.dy_pc1.names = {'dy_pc_1'};

    X.dy_pc2.Time  = Time;
    X.dy_pc2.data  = DY_PC2;
    X.dy_pc2.names = {'dy_pc_2'};

    X.dy_pc3.Time  = Time;
    X.dy_pc3.data  = DY_PC3;
    X.dy_pc3.names = {'dy_pc_3'};

    X.iv.Time  = Time;
    X.iv.data  = table2array(T_Ext(:, is_iv));
    X.iv.names = cellstr(string(T_Ext.Properties.VariableNames(is_iv)));

    X.macropc.Time  = Time;
    X.macropc.data  = table2array(T_Ext(:, is_macropc));
    X.macropc.names = cellstr(string(T_Ext.Properties.VariableNames(is_macropc)));

    X.macro.Time  = Time;
    X.macro.data  = table2array(T_Ext(:, is_macro));
    X.macro.names = cellstr(string(T_Ext.Properties.VariableNames(is_macro)));

    X.yields.Time  = Time;
    X.yields.data  = Y_all;
    X.yields.names = cellstr(string(T_Yields.Properties.VariableNames));

    y.dy.Time  = Time;
    y.dy.data  = DY_pct;
    y.dy.names = cellstr(compose('dy_%d', 1:10));

    % ---------------------------
    % 7) Metadata
    % ---------------------------
    meta = struct();
    meta.in_csv  = char(in_csv);
    meta.out_mat = char(out_mat);
    meta.X_units = 'decimals';
    meta.y_units = 'percent';
    meta.note = ['Saved with -v7. Each group has fields Time, data, names. ' ...
                 'X.fwd.data includes fwd_1 = m012 in decimals. ' ...
                 'X.slopechg.data stores 12-month slope changes in decimals. ' ...
                 'X.dy_pc.data stores expanding-window PCA scores from dy over m001..m120. ' ...
                 'X.dy_pc1, X.dy_pc2, X.dy_pc3 store individual PC scores. ' ...
                 'y.dy.data stores 12m-ahead same constant-maturity yield changes in percent for dy_1 through dy_10.'];

    % ---------------------------
    % 8) Save
    % ---------------------------
    save(out_mat, 'X', 'y', 'meta', '-v7');

end