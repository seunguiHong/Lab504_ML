clear; clc;

spec.N_factors = 5; 
spec.VERBOSE   = 0;
spec.bc        = 1;   
SAVE_PATH = "data/ACM_Signals_20.mat";

DATA_START_DATE = 197108; 
EST_START_DATE  = 199009; 
END_DATE        = 202412;

D = loadData('data/target_and_features.mat', DATA_START_DATE, END_DATE, "xr_2", ""); 

vn = string(D.Yields.Properties.VariableNames);
m_cols = vn(startsWith(vn, "m"));
m_nums = str2double(extractAfter(m_cols, "m"));

[sorted_nums, s_idx] = sort(m_nums);
valid_mask  = sorted_nums <= 120; 
target_cols = m_cols(s_idx(valid_mask));

YieldsMat = table2array(D.Yields(:, target_cols));

is_row_valid = ~any(isnan(YieldsMat), 2);
first_valid_idx = find(is_row_valid, 1, 'first');

if isempty(first_valid_idx)
    error("Critical Error: All rows contain at least one NaN.");
end

if first_valid_idx > 1
    YieldsMat = YieldsMat(first_valid_idx:end, :);
    D.Yields  = D.Yields(first_valid_idx:end, :);
end

if isdatetime(D.Yields.Time)
    DatesNum = year(D.Yields.Time)*100 + month(D.Yields.Time);
else
    DatesNum = D.Yields.Time;
end

[T, ~] = size(YieldsMat);

start_idx = find(DatesNum == EST_START_DATE, 1, 'first');

if isempty(start_idx)
    error("Estimation start date %d not found in clean data.", EST_START_DATE);
end

target_idx = (2:10) * 12; 
N = spec.N_factors;

Sig_mat = NaN(T, numel(target_idx));
Fac_mat = NaN(T, N);

parfor t = start_idx : T
    sub_Y = YieldsMat(1:t, :);
    sub_D = DatesNum(1:t);
    
    [ehprx, z_t] = ACM_Erx12(N, sub_D, sub_Y, spec);
    
    Sig_mat(t, :) = ehprx(target_idx)';
    Sig_mat = Sig_mat / 100 ;   
    Fac_mat(t, :) = z_t';
end

tgt_names = compose("xr_%d", 2:10); 
fac_names = compose("PC%d", 1:N);

SIG_ACM = [table(D.Yields.Time, 'VariableNames', {'Time'}), ...
           array2table(Sig_mat, 'VariableNames', tgt_names)];

PC_ACM  = [table(D.Yields.Time, 'VariableNames', {'Time'}), ...
           array2table(Fac_mat, 'VariableNames', fac_names)];

save(SAVE_PATH, 'SIG_ACM', 'PC_ACM');