% =========================================================================
% Date      : 2026-01-13
% Input     : raw/LW_monthly.csv, raw/Updated_LN_Macro_Factors_2025AUG.xlsx, raw/ATM_IV_40D.csv
% Output    : dataset.csv, dataset.mat
% Objective : Preprocess and merge Yield Curve (m001 format), Macro Factors, and IV data.
% Role      : data_preprocessing
% =========================================================================

function raw2dataset()
    % 1. Load and Process LW_monthly (Yield Curve)
    opts = detectImportOptions('raw/LW_monthly.csv');
    opts.DataLines = [10 Inf]; % Data starts at row 10
    opts.VariableNamingRule = 'preserve';
    
    lw_raw = readtable('raw/LW_monthly.csv', opts);
    lw_data = lw_raw(:, 1:361);
    
    % RENAME COLUMNS: Use 'm001', 'm002', ... 'm360' format
    var_names = ['Time', compose('m%03d', 1:360)];
    lw_data.Properties.VariableNames = var_names;

    for i = 2:361
        col_data = lw_data.(i);
        if iscell(col_data) || isstring(col_data)
            % Convert text/cell to double, turning non-numeric (NaN) to NaN
            lw_data.(i) = str2double(string(col_data));
        end
    end
    

    lw_data.Time = string(lw_data.Time);

    % 2. Load and Process Macro Factors
    % Note: Ensure the file extension (.xlsx or .csv) matches your actual file
    macro_raw = readtable('raw/Updated_LN_Macro_Factors_2025AUG.xlsx');
    
    % Convert 'Data' to YYYYMM string
    macro_time = datetime(macro_raw.Data); 
    macro_raw.Time = string(macro_time, 'yyyyMM');
    
    % Remove original 'Data' column and rearrange
    macro_data = removevars(macro_raw, {'Data'});
    macro_data = movevars(macro_data, 'Time', 'Before', 1);

    % 3. Load and Process ATM_IV_40D (Implied Volatility)
    iv_raw = readtable('raw/ATM_IV_40D.csv');
    
    % Convert TradeDate to YYYYMM string
    date_str = string(iv_raw.TradeDate);
    iv_raw.Time = extractBetween(date_str, 1, 6);
    
    % Sort and keep last observation per month
    iv_raw = sortrows(iv_raw, 'TradeDate');
    [~, idx_last] = unique(iv_raw.Time, 'last');
    iv_monthly = iv_raw(idx_last, :);
    
    % Select columns and Rename
    iv_data = table(iv_monthly.Time, iv_monthly.ATM_IV_5_40D, iv_monthly.ATM_IV_10_40D, iv_monthly.ATM_IV_30_40D);
    iv_data.Properties.VariableNames = {'Time', 'ATM_IV_5', 'ATM_IV_10', 'ATM_IV_30'};

    fred_file = 'fred_md_processed.csv';
    fred_data = readtable(fred_file);
    
    fred_data.Time = strtrim(string(fred_data.Time));
    
    drop_vars = ["ACOGNO","UMCSENTx","TWEXAFEGSMTHx","CP3Mx","COMPAPFFx"];
    drop_vars = intersect(drop_vars, string(fred_data.Properties.VariableNames));
    if ~isempty(drop_vars)
        fred_data = removevars(fred_data, cellstr(drop_vars));
    end

    % 4. Merge Data (Left Join on Time)
    dataset = outerjoin(lw_data, macro_data, 'Keys', 'Time', 'Type', 'left', 'MergeKeys', true);
    dataset = outerjoin(dataset, iv_data, 'Keys', 'Time', 'Type', 'left', 'MergeKeys', true);
    dataset = outerjoin(dataset, fred_data, 'Keys', 'Time', 'Type', 'left', 'MergeKeys', true);
    
    % 5. Export to CSV & MAT
    writetable(dataset, 'dataset.csv');
    save('dataset.mat', 'dataset');
end