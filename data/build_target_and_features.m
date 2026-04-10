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

    % Full annual nodes: 1y..10y
    y_annual = Y_all(:, 12:12:120);
    y_120    = Y_all(:, 1:120);

    % Inputs use 2y..10y
    y_input = y_annual(:, 2:10);

    % Targets use 1y..9y
    y_target = y_annual(:, 1:9);

    % ------------------------------------------------------------
    % slope: 2y..10y relative to 1y
    % ------------------------------------------------------------
    S = y_input - y_annual(:, 1);

    D12M_S = NaN(size(S));
    if size(S, 1) > 12
        D12M_S(13:end, :) = S(13:end, :) - S(1:end-12, :);
    end

    % ------------------------------------------------------------
    % forward: 2y..10y
    % ------------------------------------------------------------
    n_vec = 2:10;
    FWD   = y_input .* n_vec - y_annual(:, 1:9) .* (n_vec - 1);

    D12M_FWD = NaN(size(FWD));
    if size(FWD, 1) > 12
        D12M_FWD(13:end, :) = FWD(13:end, :) - FWD(1:end-12, :);
    end

    % ------------------------------------------------------------
    % 1-month yield change input for 2y..10y
    % ------------------------------------------------------------
    D1M_Y = NaN(size(y_input));
    if size(y_input, 1) > 1
        D1M_Y(2:end, :) = y_input(2:end, :) - y_input(1:end-1, :);
    end

    % ------------------------------------------------------------
    % Expanding PCA on 12-month yield changes over m001..m120
    % ------------------------------------------------------------
    T = size(y_120, 1);
    D12M_Y_120 = NaN(T, 120);
    if T > 12
        D12M_Y_120(13:end, :) = y_120(13:end, :) - y_120(1:end-12, :);
    end

    D12M_Y_PC = NaN(T, 3);
    prev_V = [];
    min_obs_pca = 24;

    for t = 13:T
        D_train = D12M_Y_120(13:t, :);
        ok_train = all(isfinite(D_train), 2);
        D_train = D_train(ok_train, :);

        if size(D_train, 1) < min_obs_pca
            continue;
        end

        x_t = D12M_Y_120(t, :);
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

        if ~isempty(prev_V)
            n_align = min(size(prev_V, 2), size(V_use, 2));
            for k = 1:n_align
                if prev_V(:, k)' * V_use(:, k) < 0
                    V_use(:, k) = -V_use(:, k);
                end
            end
        end

        score_t = (x_t - mu) * V_use;
        D12M_Y_PC(t, 1:n_keep) = score_t;

        prev_V = V_use;
    end

    D12M_Y_PC1 = D12M_Y_PC(:, 1);
    D12M_Y_PC2 = D12M_Y_PC(:, 2);
    D12M_Y_PC3 = D12M_Y_PC(:, 3);

    % ------------------------------------------------------------
    % Expanding PCA on 12-month forward changes over fwd_2..fwd_10
    % ------------------------------------------------------------
    D12M_FWD_ANN = NaN(size(FWD));
    if size(FWD, 1) > 12
        D12M_FWD_ANN(13:end, :) = FWD(13:end, :) - FWD(1:end-12, :);
    end

    D12M_FWD_PC = NaN(T, 3);
    prev_V_dfwd = [];

    for t = 13:T
        D_train = D12M_FWD_ANN(13:t, :);
        ok_train = all(isfinite(D_train), 2);
        D_train = D_train(ok_train, :);

        if size(D_train, 1) < min_obs_pca
            continue;
        end

        x_t = D12M_FWD_ANN(t, :);
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

        if ~isempty(prev_V_dfwd)
            n_align = min(size(prev_V_dfwd, 2), size(V_use, 2));
            for k = 1:n_align
                if prev_V_dfwd(:, k)' * V_use(:, k) < 0
                    V_use(:, k) = -V_use(:, k);
                end
            end
        end

        score_t = (x_t - mu) * V_use;
        D12M_FWD_PC(t, 1:n_keep) = score_t;

        prev_V_dfwd = V_use;
    end

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

    X.d1m_y.Time  = Time;
    X.d1m_y.data  = D1M_Y;
    X.d1m_y.names = cellstr(compose('d1m_y_%dy', 2:10));

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