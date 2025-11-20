%% ====================== MAIN (Intel MATLAB 2024b, hov) =======================
clear; clc; close all;
dbstop if error
rng(0);
addpath(genpath(cd));
%% -------- GLMNET path --------
glmnet_candidates = {
    '/Users/ethan_hong/Dropbox/0_Lab_504/Codes/504_ML/MATLAB/glmnet-matlab'
    '/Users/ethan_hong/Documents/MATLAB/glmnet-matlab'
    fullfile(fileparts(mfilename('fullpath')),'glmnet-matlab')
};
for c = 1:numel(glmnet_candidates)
    if isfolder(glmnet_candidates{c}), addpath(genpath(glmnet_candidates{c})); end
end
assert(~isempty(which('cvglmnet')), 'cvglmnet not found. Add glmnet-matlab to path.');
fprintf('[INFO] mexext=%s, glmnetMex=%s\n', mexext, string(~isempty(which('glmnetMex'))));

%% ------------------------------ Experiment Setting (Edit Here) -----------------------------------
DATA_FILE    = "dataset.csv";
CACHE_FILE   = "main_hov.mat";
isreload     = true;   % 첫 번째 실행만 True

% 기간/버닌(YYYYMM)
H             = 12;  
PERIOD_START  = 199109;
PERIOD_END    = 202312;
BURN_START    = 199109;
BURN_END      = 201009;

% 타깃 만기
MATURITIES = ["xr_2","xr_3","xr_5","xr_7","xr_10"];

% X (Predictor) 의 구성, 열 선택은 문자열 이름으로
spec_str   = "IV_5, IV_10, IV_30";

% 아래를 copy and paste 해서 이용
% "fwd_2, fwd_3, fwd_4, fwd_5, fwd_6, fwd_7, fwd_8, fwd_9, fwd_10";
% "F1, F2, F3, F4, F8, F1^3";
% "F1, F3, F4, F8, F1^3";
% "IV_5, IV_10, IV_30";
% "12 m, 24 m, 36 m, 48 m, 60 m, 72 m, 84 m, 96 m, 108 m, 120 m";

% Validation Fraction 결정
VAL_FRAC  = 0.30;                                 % 최근 30% 검증 
base_opts = struct('alpha',0,'standardize',1,'lambda',[]);  

%% ------------------------------- LOAD (CSV↔캐시) --------------------------
if isreload || ~isfile(CACHE_FILE)
    opts = detectImportOptions(DATA_FILE, 'TextType','string'); try, opts.VariableNamingRule='preserve'; catch, end
    T = readtable(DATA_FILE, opts);

    % Time 처리 및 기간 필터
    assert(ismember("Time", string(T.Properties.VariableNames)), "Time column required.");
    tcol = T.("Time");
    if iscellstr(tcol) || isstring(tcol)
        s = string(tcol);
        ok = ~cellfun('isempty', regexp(cellstr(s), '^\d{6}$', 'once')); assert(all(ok),'Time must be YYYYMM.');
        y = str2double(extractBefore(s,5)); m = str2double(extractAfter(s,4)); assert(all(m>=1 & m<=12));
        T.("Time") = double(y*100 + m);
    elseif isnumeric(tcol)
        v = double(tcol(:)); m = mod(v,100); assert(all(m>=1 & m<=12)); T.("Time") = v;
    else
        error('Unsupported Time column type.');
    end
    T = sortrows(T,'Time');
    T = T(T.Time>=PERIOD_START & T.Time<=PERIOD_END, :);
    assert(~isempty(T), 'No data in period.');

    % Y (xr_*)
    Y_cols = MATURITIES(ismember(MATURITIES, string(T.Properties.VariableNames)));
    assert(~isempty(Y_cols), 'MATURITIES not found in merged dataset.');
    slope_guess = regexprep(cellstr(Y_cols(:)), '^xr_', 's_');
    assert(all(ismember(slope_guess, T.Properties.VariableNames)), 'Missing slope columns: %s', ...
           strjoin(slope_guess(~ismember(slope_guess, T.Properties.VariableNames)), ', '));

    save(CACHE_FILE, 'T','Y_cols','slope_guess','PERIOD_START','PERIOD_END','-v7.3');
else
    S = load(CACHE_FILE, 'T','Y_cols','slope_guess','PERIOD_START','PERIOD_END');
    T = S.T; Y_cols = S.Y_cols; slope_guess = S.slope_guess;
end

% 워크스페이스에 병합 데이터 확인용으로 배출
assignin('base','DATASET',T);             % Input 

%% ------------------------------- BUILD ------------------------------------
Ytbl = T(:, ["Time", Y_cols]);
Time = Ytbl.Time;
Y0   = tbl2mat_no_time(Ytbl);                 % TN×J
[TN, J] = size(Y0);

Stbl = T(:, [{'Time'}, slope_guess(:)']);
S    = tbl2mat_no_time(Stbl);                 % TN×J

Xtbl = build_X_from_spec_merged(T, spec_str, [string(Y_cols), string(slope_guess')]);
X    = tbl2mat_no_time(Xtbl);                 % TN×p
p    = size(X,2);
X_VAR_NAMES = string(setdiff(Xtbl.Properties.VariableNames, {'Time'}, 'stable'));
fprintf('[INFO] Sample %d~%d (T=%d) | J=%d | p=%d\n', min(Time), max(Time), TN, J, p);

idx_burn_end = find(Time >= BURN_END, 1, 'first');   assert(~isempty(idx_burn_end), 'BURN_END not in sample.');
nOOS = TN - idx_burn_end - H;                         assert(nOOS>0, 'Not enough OOS.');

%% --------------------------- PREALLOC ------------------------------------
B     = cell(nOOS, J);        % baseline CS 계수([const; beta_s])
err_0 = zeros(nOOS, J);       % baseline
err_1 = zeros(nOOS, J);       % (1) X만
err_2 = zeros(nOOS, J);       % (2) CS-resid + X
err_3 = zeros(nOOS, J);       % (3) [s,X] penalized
err_4 = zeros(nOOS, J);       % (4) [s,X] UNpenalized
err_5 = zeros(nOOS, J);       % (5) s-only

BETA_1 = nan(nOOS, J, p);
BETA_2 = nan(nOOS, J, p);
BETA_3 = nan(nOOS, J, p);
BETA_4 = nan(nOOS, J, p);
BETA_3_SLOPE = nan(nOOS, J);
BETA_4_SLOPE = nan(nOOS, J);
BETA_5_SLOPE = nan(nOOS, J);

LAM_1 = nan(nOOS,J); LAM_2 = nan(nOOS,J); LAM_3 = nan(nOOS,J); LAM_4 = nan(nOOS,J); LAM_5 = nan(nOOS,J);
LAM1SE_1 = nan(nOOS,J); LAM1SE_2 = nan(nOOS,J); LAM1SE_3 = nan(nOOS,J); LAM1SE_4 = nan(nOOS,J); LAM1SE_5 = nan(nOOS,J);

%% ----------------------------- MAIN (5 experiments) ----------------------
tic
for i = 1:nOOS
    t  = idx_burn_end + i - 1;

    x_new = X(t+H, :);      % 테스트 입력
    s_new = S(t+H, :);      % 1×J
    y_new = Y0(t+H, :);     % 1×J

    for j = 1:J
        Xtr = X(1:t, :);
        ys  = Y0(1:t, j);
        Ss  = S(1:t, j);
        yte = y_new(j);
        ste = s_new(j);

        % (0) baseline: 동시시점 CS
        Z  = [ones(numel(Ss),1), Ss];
        Bj = Z \ ys;                           % [c; beta_s]
        yhat0 = [1, ste]*Bj;
        err_0(i,j) = yte - yhat0;
        B{i,j}     = Bj;

        % (1) Ridge on X
        [lam1, lam1se] = select_lambda_holdout(Xtr, ys, base_opts, VAL_FRAC);
        fit1 = glmnet(Xtr, ys, 'gaussian', setfield(base_opts,'lambda',lam1));
        y1   = glmnetPredict(fit1, x_new, lam1, 'response');
        err_1(i,j) = yte - y1;
        b1 = full(fit1.beta); if numel(b1)==p, BETA_1(i,j,:) = b1(:); end
        LAM_1(i,j) = lam1; LAM1SE_1(i,j) = lam1se;

        % (2) CS-residual + Ridge(X)
        res2 = ys - [ones(numel(Ss),1), Ss]*Bj;
        [lam2, lam2se] = select_lambda_holdout(Xtr, res2, base_opts, VAL_FRAC);
        fit2 = glmnet(Xtr, res2, 'gaussian', setfield(base_opts,'lambda',lam2));
        add2 = glmnetPredict(fit2, x_new, lam2, 'response');
        y2   = [1, ste]*Bj + add2;
        err_2(i,j) = yte - y2;
        b2 = full(fit2.beta); if numel(b2)==p, BETA_2(i,j,:) = b2(:); end
        LAM_2(i,j) = lam2; LAM1SE_2(i,j) = lam2se;

        % (3) Ridge on [slope, X] (penalized slope)
        X3 = [Ss, Xtr];  opts3 = base_opts; opts3.penalty_factor = ones(1, size(X3,2));
        [lam3, lam3se] = select_lambda_holdout(X3, ys, opts3, VAL_FRAC);
        fit3 = glmnet(X3, ys, 'gaussian', setfield(opts3,'lambda',lam3));
        y3   = glmnetPredict(fit3, [ste, x_new], lam3, 'response');
        err_3(i,j) = yte - y3;
        b3 = full(fit3.beta); if numel(b3)==(1+p), BETA_3_SLOPE(i,j)=b3(1); BETA_3(i,j,:)=b3(2:end); end
        LAM_3(i,j) = lam3; LAM1SE_3(i,j) = lam3se;

        % (4) Ridge on [slope, X] (UNpenalized slope)
        X4 = [Ss, Xtr];  opts4 = base_opts; opts4.penalty_factor = [0, ones(1,size(Xtr,2))];
        [lam4, lam4se] = select_lambda_holdout(X4, ys, opts4, VAL_FRAC);
        fit4 = glmnet(X4, ys, 'gaussian', setfield(opts4,'lambda',lam4));
        y4   = glmnetPredict(fit4, [ste, x_new], lam4, 'response');
        err_4(i,j) = yte - y4;
        b4 = full(fit4.beta); if numel(b4)==(1+p), BETA_4_SLOPE(i,j)=b4(1); BETA_4(i,j,:)=b4(2:end); end
        LAM_4(i,j) = lam4; LAM1SE_4(i,j) = lam4se;

        % (5) Ridge slope-only
        X5 = Ss;  y5 = ys;  opts5 = base_opts; opts5.penalty_factor = 1;
        [lam5, lam5se] = select_lambda_holdout(X5, y5, opts5, VAL_FRAC);
        fit5 = glmnet(X5, y5, 'gaussian', setfield(opts5,'lambda',lam5));
        y5p  = glmnetPredict(fit5, ste, lam5, 'response');
        err_5(i,j) = yte - y5p;
        b5 = full(fit5.beta); if ~isempty(b5), BETA_5_SLOPE(i,j) = b5(1); end
        LAM_5(i,j) = lam5; LAM1SE_5(i,j) = lam5se;
    end
end
toc

%% ----------------------------- REPORT ------------------------------------
R2 = @(e) 1 - sum(e.^2,1)./sum(err_0.^2,1);
R2_tbl = array2table([R2(err_1); R2(err_2); R2(err_3); R2(err_4); R2(err_5)], ...
    'VariableNames', cellstr(Y_cols), ...
    'RowNames', {'(1) X ridge','(2)CS-residual + Ridge(X only)','(3)Ridge+penalized slope','(4)Ridge+unpenalized slope','(5)s-only'});
disp(R2_tbl);

% ---- 워크스페이스로 내보내기 ----
assignin('base','PREDICTORS',X_VAR_NAMES);
assignin('base','LAM_1',LAM_1);       assignin('base','LAM_2',LAM_2);
assignin('base','LAM_3',LAM_3);       assignin('base','LAM_4',LAM_4); assignin('base','LAM_5',LAM_5);
assignin('base','LAM1SE_1',LAM1SE_1); assignin('base','LAM1SE_2',LAM1SE_2);
assignin('base','LAM1SE_3',LAM1SE_3); assignin('base','LAM1SE_4',LAM1SE_4); assignin('base','LAM1SE_5',LAM1SE_5);
assignin('base','ERR_BASE',err_0);
assignin('base','ERR_1',err_1); assignin('base','ERR_2',err_2);
assignin('base','ERR_3',err_3); assignin('base','ERR_4',err_4); assignin('base','ERR_5',err_5);

%% ============================ HELPERS ====================================
function M = tbl2mat_no_time(T)
    if any(strcmp('Time', T.Properties.VariableNames))
        try, T = removevars(T, 'Time'); catch
            idx = find(strcmp('Time', T.Properties.VariableNames), 1);
            T = T(:, setdiff(1:width(T), idx));
        end
    end
    M = double(table2array(T));
end

function [lambda_sel, lambda_1se] = select_lambda_holdout(X, y, opts, val_frac)
% 시간순 holdout으로 λ 선택. opts.lambda가 []면 glmnet 자동경로 사용.
% 반환: lambda_sel(=min MSE), lambda_1se(=1-SE rule에서 가장 큰 λ)
    n  = size(X,1);
    nv = max(1, floor(val_frac * n));
    ntr = n - nv;  assert(ntr >= 5, 'Too few training obs for holdout.');
    Xtr = X(1:ntr,:);  ytr = y(1:ntr);
    Xva = X(ntr+1:end,:);  yva = y(ntr+1:end);

    if isempty(opts.lambda)
        fit_master  = glmnet(Xtr, ytr, 'gaussian', rmfield_if(opts,'lambda'));
        lambda_grid = fit_master.lambda(:);
    else
        lambda_grid = opts.lambda(:);
    end

    fit  = glmnet(Xtr, ytr, 'gaussian', setfield(opts,'lambda',lambda_grid)); %#ok<SFLD>
    pred = glmnetPredict(fit, Xva, lambda_grid, 'response');   % |V|×|λ|
    err  = (yva - pred).^2;                    % V×|λ|
    mse  = mean(err, 1);                       % 1×|λ|
    se   = std(err, 0, 1) / sqrt(size(err,1)); % 1×|λ|

    [mmin, ix_min] = min(mse);
    lambda_sel = lambda_grid(ix_min);

    tol = mmin + se(ix_min);
    idx_ok = find(mse <= tol);
    if ~isempty(idx_ok)
        [~, ix_maxlam] = max(lambda_grid(idx_ok));   % 가장 큰 λ
        lambda_1se = lambda_grid(idx_ok(ix_maxlam));
    else
        lambda_1se = lambda_sel;
    end
end

function S = rmfield_if(S, fname)
    if isfield(S, fname), S = rmfield(S, fname); end
end

function Xtbl = build_X_from_spec_merged(T, spec_str, reserved_list)
% spec_str 예: "fwd_2 + fwd_3, iv_10"
% - 데이터셋에 실제로 존재하는 열만 허용
% - '+' 또는 ',' 로 구분
% - 존재하지 않으면 error
% - reserved(Time, Y, slope)은 자동 제외

    vn   = string(T.Properties.VariableNames);
    reserved = lower(["Time", string(reserved_list(:)')]);

    % 토큰화
    raw = regexp(char(spec_str), '\s*[+,]\s*', 'split');
    want = unique(string(strtrim(raw)),"stable");

    % 선택 확인
    selected = strings(0,1);
    missing  = strings(0,1);
    for w = want(:)'
        if w==""; continue; end
        idx = find(strcmpi(vn, w), 1, 'first');
        if ~isempty(idx)
            cand = vn(idx);
            if any(strcmpi(reserved, cand)), continue; end
            vcol = T{:, idx};
            if ~(isnumeric(vcol) || islogical(vcol))
                fprintf('[WARN] Non-numeric skipped: %s\n', cand);
                continue;
            end
            selected(end+1,1) = cand;
        else
            missing(end+1,1) = w;
        end
    end

    if ~isempty(missing)
        error('Unknown columns in spec_str: %s', strjoin(cellstr(missing), ', '));
    end
    if isempty(selected)
        error('No valid predictors selected.');
    end

    Xtbl = T(:, [{'Time'}, cellstr(selected)']);
end