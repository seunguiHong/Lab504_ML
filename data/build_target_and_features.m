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

    y_annual = Y_all(:, 12:12:120);
    y_120    = Y_all(:, 1:120);

    S = y_annual(:, 2:end) - y_annual(:, 1);

    S_chg = NaN(size(S));
    if size(S, 1) > 12
        S_chg(13:end, :) = S(13:end, :) - S(1:end-12, :);
    end

    n_vec = 2:10;
    FWD   = y_annual(:, 2:end) .* n_vec - y_annual(:, 1:end-1) .* (n_vec - 1);
    rf12  = y_annual(:, 1);

    FWD_chg = NaN(size(FWD));
    if size(FWD, 1) > 12
        FWD_chg(13:end, :) = FWD(13:end, :) - FWD(1:end-12, :);
    end

    T = size(y_120, 1);
    DY_120 = NaN(T, 120);
    if T > 12
        DY_120(13:end, :) = y_120(13:end, :) - y_120(1:end-12, :);
    end

    DY_PC = NaN(T, 3);
    prev_V = [];
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

    % ------------------------------------------------------------
    % External blocks
    % ------------------------------------------------------------
    T_Ext  = raw(:, ~y_mask & vn ~= "Time");
    ext_vn = string(T_Ext.Properties.VariableNames);

    % keep old behavior
    is_iv      = startsWith(ext_vn, "ATM_IV_");
    is_macropc = ~cellfun('isempty', regexp(ext_vn, '^F\d+$', 'once'));
    is_macro   = ~(is_iv | is_macropc);

    DY_dec = NaN(T, 9);
    if T > 12
        DY_dec(1:T-12, :) = y_annual(13:T, 1:9) - y_annual(1:T-12, 1:9);
    end
    DY_pct = 100 * DY_dec;

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

    X.fwdchg.Time  = Time;
    X.fwdchg.data  = FWD_chg;
    X.fwdchg.names = cellstr(compose('dfwd_%d', 2:10));

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

    % keep old behavior
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

% ============================================================
% helper: block creator
% ============================================================
function B = make_block(Time, data, names)
    B = struct();
    B.Time  = Time;
    B.data  = data;
    B.names = cellstr(names(:)');
end

% ============================================================
% helper: FRED-MD group mapping
% based on variable names in your 2025-12-MD.csv / fred_md_processed.csv
% ============================================================
function G = fred_md_group_definitions()

    G = struct();

    % 1. Output / income
    G.output = string({ ...
        'RPI','W875RX1','DPCERA3M086SBEA','CMRMTSPLx','RETAILx', ...
        'INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD', ...
        'IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222S','IPFUELS', ...
        'CUMFNS', ...
        'DTCOLNVHFNM','DTCTHFNM','INVEST' ...
    });

    % 2. Labor market
    G.labor = string({ ...
        'HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN', ...
        'UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx', ...
        'PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP', ...
        'NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT', ...
        'CES0600000007','AWOTMAN','AWHMAN', ...
        'CES0600000008','CES2000000008','CES3000000008' ...
    });

    % 3. Housing
    G.housing = string({ ...
        'HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW', ...
        'PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW' ...
    });

    % 4. Orders / inventories
    G.orders = string({ ...
        'ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx' ...
    });

    % 5. Money / credit
    G.money = string({ ...
        'M1SL','M2SL','M2REAL','BOGMBASE','TOTRESNS','NONBORRES', ...
        'BUSLOANS','REALLN','NONREVSL','CONSPI' ...
    });

    % 6. Rates / FX / bond market / financial conditions
    G.ratesfx = string({ ...
        'FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA', ...
        'COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM', ...
        'AAAFFM','BAAFFM','TWEXAFEGSMTHx','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx' ...
    });

    % 7. Prices
    G.prices = string({ ...
        'WPSFD49207','WPSFD49502','WPSID61','WPSID62','OILPRICEx','PPICMM', ...
        'CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC','CUSR0000SAD', ...
        'CUSR0000SAS','CPIULFSL','CUSR0000SA0L2','CUSR0000SA0L5','PCEPI', ...
        'DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA' ...
    });

    % 8. Stock market / sentiment / volatility
    G.stock = string({ ...
        'S&P 500','S&P div yield','S&P PE ratio','UMCSENTx','VIXCLSx' ...
    });

end