function build_target_and_features(in_csv, out_mat)

    if nargin < 1 || isempty(in_csv),  in_csv = 'dataset.csv'; end
    if nargin < 2 || isempty(out_mat), out_mat = 'target_and_features.mat'; end

    opts = detectImportOptions(in_csv, 'VariableNamingRule', 'preserve');
    raw  = readtable(in_csv, opts);

    vn = string(raw.Properties.VariableNames);
    if ~ismember("Time", vn)
        error("Missing required column: Time");
    end

    Time = raw.Time(:);

    % ------------------------------------------------------------
    % Yield block
    % ------------------------------------------------------------
    y_mask = ~cellfun('isempty', regexp(vn, '^m\d{3}$', 'once'));
    if sum(y_mask) < 120
        error("Insufficient yield columns: need at least m001..m120.");
    end

    T_Yields = raw(:, y_mask);
    Y_all    = table2array(T_Yields) / 100;

    if size(Y_all, 2) < 120
        error("Yield matrix has fewer than 120 monthly maturities.");
    end

    T = size(Y_all, 1);

    % Full annual nodes: 1y..10y
    y_annual = Y_all(:, 12:12:120);

    % Inputs use 2y..10y
    y_input = y_annual(:, 2:10);

    % Targets use 1y..9y
    y_target = y_annual(:, 1:9);

    % ------------------------------------------------------------
    % slope: 2y..10y relative to 1y
    % ------------------------------------------------------------
    S = y_input - y_annual(:, 1);

    D12M_S = NaN(size(S));
    if T > 12
        D12M_S(13:end, :) = S(13:end, :) - S(1:end-12, :);
    end

    % ------------------------------------------------------------
    % forward: 2y..10y
    % ------------------------------------------------------------
    n_vec = 2:10;
    FWD   = y_input .* n_vec - y_annual(:, 1:9) .* (n_vec - 1);

    D12M_FWD = NaN(size(FWD));
    if T > 12
        D12M_FWD(13:end, :) = FWD(13:end, :) - FWD(1:end-12, :);
    end

    % ------------------------------------------------------------
    % 12-month yield change input for 2y..10y
    % ------------------------------------------------------------
    D12M_Y = NaN(size(y_input));
    if T > 12
        D12M_Y(13:end, :) = y_input(13:end, :) - y_input(1:end-12, :);
    end

    % ------------------------------------------------------------
    % 1-month yield change input for 2y..10y
    % ------------------------------------------------------------
    D1M_Y = NaN(size(y_input));
    if T > 1
        D1M_Y(2:end, :) = y_input(2:end, :) - y_input(1:end-1, :);
    end

    % ------------------------------------------------------------
    % Expanding PCA on 12-month yield changes over annual maturities 1y..10y
    % ------------------------------------------------------------
    D12M_Y_ANN = NaN(T, 10);
    if T > 12
        D12M_Y_ANN(13:end, :) = y_annual(13:end, :) - y_annual(1:end-12, :);
    end

    min_obs_pca = 24;
    D12M_Y_PC = recursive_pca_scores(D12M_Y_ANN, 3, min_obs_pca);

    D12M_Y_PC1 = D12M_Y_PC(:, 1);
    D12M_Y_PC2 = D12M_Y_PC(:, 2);
    D12M_Y_PC3 = D12M_Y_PC(:, 3);

    % ------------------------------------------------------------
    % Expanding PCA on 12-month forward changes over annual forwards 2y..10y
    % ------------------------------------------------------------
    D12M_FWD_PC = recursive_pca_scores(D12M_FWD, 3, min_obs_pca);

    D12M_FWD_PC1 = D12M_FWD_PC(:, 1);
    D12M_FWD_PC2 = D12M_FWD_PC(:, 2);
    D12M_FWD_PC3 = D12M_FWD_PC(:, 3);

    % ------------------------------------------------------------
    % External blocks
    % ------------------------------------------------------------
    T_Ext  = raw(:, ~y_mask & vn ~= "Time");
    ext_vn = string(T_Ext.Properties.VariableNames);

    is_iv      = startsWith(ext_vn, "ATM_IV_");
    is_macropc = ~cellfun('isempty', regexp(ext_vn, '^F\d+$', 'once'));
    is_macro   = ~(is_iv | is_macropc);

    % ------------------------------------------------------------
    % Target: keep existing naming rule dy_1..dy_9
    % ------------------------------------------------------------
    DY_dec = NaN(T, 9);
    if T > 12
        DY_dec(1:T-12, :) = y_target(13:T, :) - y_target(1:T-12, :);
    end
    DY_pct = 100 * DY_dec;

    % ------------------------------------------------------------
    % Target: one-year excess returns rx_2..rx_10
    % rx_{t+12}^{(n)} = n*y_t^{(n)}
    %                 - (n-1)*y_{t+12}^{(n-1)}
    %                 - y_t^{(1)}
    % n = 2,...,10
    % Units: percent
    % ------------------------------------------------------------
    RX_dec = NaN(T, 9);
    if T > 12
        n_rx = 2:10;

        RX_dec(1:T-12, :) = ...
            y_annual(1:T-12, 2:10) .* n_rx ...
            - y_annual(13:T, 1:9) .* (n_rx - 1) ...
            - y_annual(1:T-12, 1);
    end
    RX_pct = 100 * RX_dec;

    % ------------------------------------------------------------
    % Canonical recursive Cochrane-Piazzesi factor
    % ------------------------------------------------------------
    % Canonical CP regression:
    %
    %   avg_rx_{t+12}^{2:5}
    %       = gamma_0
    %       + gamma_1 y_t^{(1)}
    %       + gamma_2 f_t^{(2)}
    %       + gamma_3 f_t^{(3)}
    %       + gamma_4 f_t^{(4)}
    %       + gamma_5 f_t^{(5)}
    %       + error_{t+12}
    %
    % Predictors:
    %   [y1, fwd_2, fwd_3, fwd_4, fwd_5]
    %
    % Target:
    %   mean(rx_2, rx_3, rx_4, rx_5)
    %
    % Leakage control:
    %   row s enters the CP regression at forecast-origin t only if
    %   s + horizon <= t.
    %
    % Units:
    %   y_annual : decimals
    %   FWD      : decimals
    %   RX_dec   : decimals
    %   CP       : decimals
    % ------------------------------------------------------------
    horizon = 12;
    min_obs_cp = 120;

    CP_X = [y_annual(:, 1), FWD(:, 1:4)];
    avg_rx_cp_dec = mean(RX_dec(:, 1:4), 2, 'omitnan');

    CP = NaN(T, 1);

    for t = 1:T

        train_end = t - horizon;

        if train_end < min_obs_cp
            continue;
        end

        idx_train = (1:train_end)';

        ok_train = all(isfinite(CP_X(idx_train, :)), 2) & ...
                   isfinite(avg_rx_cp_dec(idx_train));

        if sum(ok_train) < min_obs_cp
            continue;
        end

        Xtrain = CP_X(idx_train(ok_train), :);
        ytrain = avg_rx_cp_dec(idx_train(ok_train));

        Ztrain = [ones(size(Xtrain, 1), 1), Xtrain];

        b = Ztrain \ ytrain;

        if all(isfinite(CP_X(t, :)))
            CP(t) = [1, CP_X(t, :)] * b;
        end
    end

    X = struct();
    y = struct();

    X.time = Time;
    y.time = Time;

    X.slope.Time  = Time;
    X.slope.data  = S;
    X.slope.names = cellstr(compose('slope_%dy', 2:10));

    X.d12m_slope.Time  = Time;
    X.d12m_slope.data  = D12M_S;
    X.d12m_slope.names = cellstr(compose('d12m_slope_%dy', 2:10));

    X.fwd.Time  = Time;
    X.fwd.data  = FWD;
    X.fwd.names = cellstr(compose('fwd_%dy', 2:10));

    X.d12m_fwd.Time  = Time;
    X.d12m_fwd.data  = D12M_FWD;
    X.d12m_fwd.names = cellstr(compose('d12m_fwd_%dy', 2:10));

    X.d12m_y.Time  = Time;
    X.d12m_y.data  = D12M_Y;
    X.d12m_y.names = cellstr(compose('d12m_y_%dy', 2:10));

    X.d1m_y.Time  = Time;
    X.d1m_y.data  = D1M_Y;
    X.d1m_y.names = cellstr(compose('d1m_y_%dy', 2:10));

    X.cp.Time  = Time;
    X.cp.data  = CP;
    X.cp.names = {'cp'};

    X.d12m_y_pc.Time  = Time;
    X.d12m_y_pc.data  = D12M_Y_PC;
    X.d12m_y_pc.names = {'d12m_y_pc1', 'd12m_y_pc2', 'd12m_y_pc3'};

    X.d12m_y_pc1.Time  = Time;
    X.d12m_y_pc1.data  = D12M_Y_PC1;
    X.d12m_y_pc1.names = {'d12m_y_pc1'};

    X.d12m_y_pc2.Time  = Time;
    X.d12m_y_pc2.data  = D12M_Y_PC2;
    X.d12m_y_pc2.names = {'d12m_y_pc2'};

    X.d12m_y_pc3.Time  = Time;
    X.d12m_y_pc3.data  = D12M_Y_PC3;
    X.d12m_y_pc3.names = {'d12m_y_pc3'};

    X.d12m_fwd_pc.Time  = Time;
    X.d12m_fwd_pc.data  = D12M_FWD_PC;
    X.d12m_fwd_pc.names = {'d12m_fwd_pc1', 'd12m_fwd_pc2', 'd12m_fwd_pc3'};

    X.d12m_fwd_pc1.Time  = Time;
    X.d12m_fwd_pc1.data  = D12M_FWD_PC1;
    X.d12m_fwd_pc1.names = {'d12m_fwd_pc1'};

    X.d12m_fwd_pc2.Time  = Time;
    X.d12m_fwd_pc2.data  = D12M_FWD_PC2;
    X.d12m_fwd_pc2.names = {'d12m_fwd_pc2'};

    X.d12m_fwd_pc3.Time  = Time;
    X.d12m_fwd_pc3.data  = D12M_FWD_PC3;
    X.d12m_fwd_pc3.names = {'d12m_fwd_pc3'};

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
    y.dy.names = cellstr(compose('dy_%d', 1:9));

    y.rx.Time  = Time;
    y.rx.data  = RX_pct;
    y.rx.names = cellstr(compose('rx_%d', 2:10));

    % ------------------------------------------------------------
    % Macro group blocks
    % ------------------------------------------------------------
    macro_names = string(X.macro.names);
    G = fred_md_group_definitions();

    is_output  = ismember(macro_names, G.output);
    is_labor   = ismember(macro_names, G.labor);
    is_housing = ismember(macro_names, G.housing);
    is_orders  = ismember(macro_names, G.orders);
    is_money   = ismember(macro_names, G.money);
    is_ratesfx = ismember(macro_names, G.ratesfx);
    is_prices  = ismember(macro_names, G.prices);
    is_stock   = ismember(macro_names, G.stock);

    is_any_group = is_output | is_labor | is_housing | is_orders | ...
                   is_money | is_ratesfx | is_prices | is_stock;
    is_other = ~is_any_group;

    X.macro_output  = make_block(Time, X.macro.data(:, is_output),  macro_names(is_output));
    X.macro_labor   = make_block(Time, X.macro.data(:, is_labor),   macro_names(is_labor));
    X.macro_housing = make_block(Time, X.macro.data(:, is_housing), macro_names(is_housing));
    X.macro_orders  = make_block(Time, X.macro.data(:, is_orders),  macro_names(is_orders));
    X.macro_money   = make_block(Time, X.macro.data(:, is_money),   macro_names(is_money));
    X.macro_ratesfx = make_block(Time, X.macro.data(:, is_ratesfx), macro_names(is_ratesfx));
    X.macro_prices  = make_block(Time, X.macro.data(:, is_prices),  macro_names(is_prices));
    X.macro_stock   = make_block(Time, X.macro.data(:, is_stock),   macro_names(is_stock));
    X.macro_other   = make_block(Time, X.macro.data(:, is_other),   macro_names(is_other));

    meta = struct();
    meta.in_csv  = char(in_csv);
    meta.out_mat = char(out_mat);
    meta.X_units = 'decimals';
    meta.y_units = 'percent';

    meta.cp_note = ...
        'X.cp is the canonical recursive Cochrane-Piazzesi factor. It is estimated from avg rx_2..rx_5 on [y1, fwd_2, fwd_3, fwd_4, fwd_5] using only rows s with s+12 <= t. CP is stored in decimals.';

    meta.d12m_y_note = ...
        'X.d12m_y contains trailing 12-month yield changes for annual maturities 2y..10y, stored in decimals.';

    meta.d12m_y_pc_note = ...
        'X.d12m_y_pc contains expanding PCA scores from trailing 12-month yield changes over annual maturities 1y..10y. PCA loadings are estimated recursively using observations through t, and signs are aligned to the previous available loading. Scores are stored in decimals.';

    meta.d12m_fwd_pc_note = ...
        'X.d12m_fwd_pc contains expanding PCA scores from trailing 12-month forward-rate changes over annual forwards 2y..10y. PCA loadings are estimated recursively using observations through t, and signs are aligned to the previous available loading. Scores are stored in decimals.';

    meta.pca_min_obs = min_obs_pca;

    meta.macro_group_names = { ...
        'macro_output', ...
        'macro_labor', ...
        'macro_housing', ...
        'macro_orders', ...
        'macro_money', ...
        'macro_ratesfx', ...
        'macro_prices', ...
        'macro_stock', ...
        'macro_other'};

    save(out_mat, 'X', 'y', 'meta', '-v7');

end

function PC = recursive_pca_scores(D, n_pc, min_obs)

    T = size(D, 1);
    PC = NaN(T, n_pc);
    prev_V = [];

    for t = 1:T

        D_train = D(1:t, :);
        ok_train = all(isfinite(D_train), 2);
        D_train = D_train(ok_train, :);

        if size(D_train, 1) < min_obs
            continue;
        end

        x_t = D(t, :);
        if ~all(isfinite(x_t))
            continue;
        end

        mu = mean(D_train, 1);
        D_center = D_train - mu;

        [~, ~, V] = svd(D_center, 'econ');

        n_keep = min(n_pc, size(V, 2));
        if n_keep < 1
            continue;
        end

        V_use = V(:, 1:n_keep);

        if ~isempty(prev_V)
            n_align = min(size(prev_V, 2), size(V_use, 2));
            for k = 1:n_align
                if prev_V(:, k)' * V_use(:, k) < 0
                    V_use(:, k) = -V_use(:, k);
                end
            end
        end

        score_t = (x_t - mu) * V_use;
        PC(t, 1:n_keep) = score_t;

        prev_V = V_use;
    end

end

function B = make_block(Time, data, names)

    B = struct();
    B.Time  = Time;
    B.data  = data;
    B.names = cellstr(names(:)');

end

function G = fred_md_group_definitions()

    G = struct();

    G.output = string({ ...
        'RPI','W875RX1','DPCERA3M086SBEA','CMRMTSPLx','RETAILx', ...
        'INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD', ...
        'IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222S','IPFUELS', ...
        'CUMFNS', ...
        'DTCOLNVHFNM','DTCTHFNM','INVEST' ...
    });

    G.labor = string({ ...
        'HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN', ...
        'UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx', ...
        'PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP', ...
        'NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT', ...
        'CES0600000007','AWOTMAN','AWHMAN', ...
        'CES0600000008','CES2000000008','CES3000000008' ...
    });

    G.housing = string({ ...
        'HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW', ...
        'PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW' ...
    });

    G.orders = string({ ...
        'ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx' ...
    });

    G.money = string({ ...
        'M1SL','M2SL','M2REAL','BOGMBASE','TOTRESNS','NONBORRES', ...
        'BUSLOANS','REALLN','NONREVSL','CONSPI' ...
    });

    G.ratesfx = string({ ...
        'FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA', ...
        'COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM', ...
        'AAAFFM','BAAFFM','TWEXAFEGSMTHx','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx' ...
    });

    G.prices = string({ ...
        'WPSFD49207','WPSFD49502','WPSID61','WPSID62','OILPRICEx','PPICMM', ...
        'CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC','CUSR0000SAD', ...
        'CUSR0000SAS','CPIULFSL','CUSR0000SA0L2','CUSR0000SA0L5','PCEPI', ...
        'DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA' ...
    });

    G.stock = string({ ...
        'S&P 500','S&P div yield','S&P PE ratio','UMCSENTx','VIXCLSx' ...
    });

end