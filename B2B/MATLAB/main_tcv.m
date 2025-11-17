%% ====================== MAIN (Intel MATLAB 2024b, tcv) =======================
clear; clc; close all;
dbstop if error
rng(1);
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
assert(~isempty(which('cvglmnet')) && ~isempty(which('glmnet')), ...
    'glmnet-matlab functions not found on path.');
fprintf('[INFO] mexext=%s, glmnetMex=%s\n', mexext, string(~isempty(which('glmnetMex'))));

%% ------------------------------ Experiment Setting (Edit Here) -----------------------------------
DATA_FILE   = "dataset.csv";    
CACHE_FILE  = "main_tcv_cache.mat";   % <--- ncv → tcv
isreload    = true;                    % 첫 실행 true 

% 기간/버닌(YYYYMM)
H             = 12;
PERIOD_START  = 199109;
PERIOD_END    = 202312;
BURN_START    = 199109;
BURN_END      = 201009;

% 타깃 만기
MATURITIES = ["xr_2","xr_5","xr_7","xr_10"];

% X (Predictor)의 구성, 열 선택은 문자열 이름으로
spec_str   = "CP";

% 아래를 copy and paste 해서 이용
% "fwd_2, fwd_3, fwd_4, fwd_5, fwd_6, fwd_7, fwd_8, fwd_9, fwd_10";
% "F1, F2, F3, F4, F8, F1^3";
% "F1, F3, F4, F8, F1^3";
% "IV_5, IV_10, IV_30";
% "12 m, 24 m, 36 m, 48 m, 60 m, 72 m, 84 m, 96 m, 108 m, 120 m";

% GLMNET/검증 옵션 
K = 10;  % 블록 개수
base_opts = struct('alpha',0,'standardize',1,'lambda',[]);  % ridge ([]면 자동 경로)

%% ============================ LOAD (Merged) ================================
if isreload || ~isfile(CACHE_FILE)
    opts = detectImportOptions(DATA_FILE, 'TextType','string'); try, opts.VariableNamingRule='preserve'; catch, end
    T_raw = readtable(DATA_FILE, opts);

    % Time → YYYYMM(double)
    assert(ismember("Time", string(T_raw.Properties.VariableNames)), "Time column required.");
    tcol = T_raw.("Time");
    if iscellstr(tcol) || isstring(tcol)
        s = string(tcol);
        ok = ~cellfun('isempty', regexp(cellstr(s), '^\d{6}$', 'once')); assert(all(ok),'Time must be YYYYMM.');
        y = str2double(extractBefore(s,5)); m = str2double(extractAfter(s,4)); assert(all(m>=1 & m<=12));
        T_raw.("Time") = double(y*100 + m);
    elseif isnumeric(tcol)
        v = double(tcol(:)); m = mod(v,100); assert(all(m>=1 & m<=12)); T_raw.("Time") = v;
    else
        error('Unsupported Time column type.');
    end
    T_raw = sortrows(T_raw,'Time');
    save(CACHE_FILE,'T_raw','-v7.3');
else
    S0 = load(CACHE_FILE,'T_raw');
    T_raw = S0.T_raw;
end

% 기간 필터
T = T_raw(T_raw.Time>=PERIOD_START & T_raw.Time<=PERIOD_END, :);
assert(~isempty(T), 'No data in period.');

% Y (xr_*) 및 slope 매칭(s_k)
Y_cols = MATURITIES(ismember(MATURITIES, string(T.Properties.VariableNames)));
assert(~isempty(Y_cols), 'MATURITIES not found in merged dataset.');
slope_guess = regexprep(cellstr(Y_cols(:)), '^xr_', 's_');
assert(all(ismember(slope_guess, T.Properties.VariableNames)), 'Missing slope columns: %s', ...
       strjoin(slope_guess(~ismember(slope_guess, T.Properties.VariableNames)), ', '));

Ytbl = T(:, ["Time", Y_cols]);
Stbl = T(:, [{'Time'}, slope_guess(:)']);

Time = Ytbl.Time;
Y0   = tbl2mat_no_time(Ytbl);                 % TN×J
S    = tbl2mat_no_time(Stbl);                 % TN×J
[TN, J] = size(Y0);

% X: spec_str에 따라 선택(중복/비수치/예약 컬럼 제거)
Xtbl = build_X_from_spec_merged(T, spec_str, [string(Y_cols), string(slope_guess')]);
X    = tbl2mat_no_time(Xtbl);                 % TN×p
p    = size(X,2);

% 프레딕터 이름(표시/워크스페이스)
X_names     = setdiff(Xtbl.Properties.VariableNames, {'Time'}, 'stable');   % cellstr
X_VAR_NAMES = string(X_names);  % holdout 버전과 호환

fprintf('[INFO] Sample %d~%d (T=%d) | J=%d | p=%d\n', min(Time), max(Time), TN, J, p);

% 버닌 및 OOS
idx_burn_end = find(Time >= BURN_END, 1, 'first');   assert(~isempty(idx_burn_end), 'BURN_END not in sample.');
nOOS = TN - idx_burn_end - H;                         assert(nOOS>0, 'Not enough OOS observations.');

% 워크스페이스 확인용
assignin('base','DATASET',T);
assignin('base','Xtbl',Xtbl);
assignin('base','X_names',X_names);
assignin('base','X_VAR_NAMES',X_VAR_NAMES);

%% --------------------------- PREALLOC (원본 유지) --------------------------
B     = cell(nOOS, J);   % baseline CS 계수([const; beta_slope]) 또는 상수-only
err_0 = zeros(nOOS, J);  % baseline (동시시점 CS 또는 상수-only)
err_1 = zeros(nOOS, J);  % Ridge on X (확장창 블록 CV)
err_2 = zeros(nOOS, J);  % CS-residual + Ridge(X only)
err_3 = zeros(nOOS, J);  % Ridge with penalized slope [s_j, X]
err_4 = zeros(nOOS, J);  % Ridge with *unpenalized* slope
err_5 = zeros(nOOS, J);  % slope-only (Ridge)

% ---- 계수 저장용 (각 셀에 [intercept; betas]) ----
Beta_1 = cell(nOOS, J);
Beta_2 = cell(nOOS, J);
Beta_3 = cell(nOOS, J);
Beta_4 = cell(nOOS, J);
Beta_5 = cell(nOOS, J);

% ---- λ 경로/통계 로깅 ----
LAMPATH_1 = cell(nOOS,J);  CVM_1 = cell(nOOS,J);  CVSD_1 = cell(nOOS,J);
LAMPATH_2 = cell(nOOS,J);  CVM_2 = cell(nOOS,J);  CVSD_2 = cell(nOOS,J);
LAMPATH_3 = cell(nOOS,J);  CVM_3 = cell(nOOS,J);  CVSD_3 = cell(nOOS,J);
LAMPATH_4 = cell(nOOS,J);  CVM_4 = cell(nOOS,J);  CVSD_4 = cell(nOOS,J);
LAMPATH_5 = cell(nOOS,J);  CVM_5 = cell(nOOS,J);  CVSD_5 = cell(nOOS,J);
LAM_1 = nan(nOOS,J);  LAM1SE_1 = nan(nOOS,J);
LAM_2 = nan(nOOS,J);  LAM1SE_2 = nan(nOOS,J);
LAM_3 = nan(nOOS,J);  LAM1SE_3 = nan(nOOS,J);
LAM_4 = nan(nOOS,J);  LAM1SE_4 = nan(nOOS,J);
LAM_5 = nan(nOOS,J);  LAM1SE_5 = nan(nOOS,J);

%% ----------------------------- MAIN (원본 NCV) ----------------------------
% 학습/검증: (X_s, y_s) 동시시점
% 테스트:   (X_{t+H}, y_{t+H}) 동시시점
tic
for i = 1:nOOS
    t  = idx_burn_end + i - 1;      % 학습/검증 윈도 마지막 시점

    for j = 1:J
        % -------------------- 공통 학습/검증 세트 --------------------
        Xtr_full = X(1:t, :);              % n × p
        y_full   = Y0(1:t, j);             % n × 1
        Sj_full  = S(1:t, j);              % n × 1

        % 테스트 입력/정답 (동시시점, t+H)
        x_new    = X(t+H, :);              % 1 × p
        sj_new   = S(t+H, j);              % 1 × 1
        y_true   = Y0(t+H, j);             % 1 × 1

        % NaN 행 제거(학습/검증)
        m1 = all(~isnan(Xtr_full),2) & ~isnan(y_full);
        m2 = m1 & ~isnan(Sj_full);

        X1 = Xtr_full(m1,:);    y1 = y_full(m1);     % 실험1용 (slope 미사용)
        Xs = Xtr_full(m2,:);    ys = y_full(m2);     % 2~5용 (slope 사용)
        Ss = Sj_full(m2);

        % ---------- (0) 동시시점 Campbell–Shiller baseline ----------
        if ~isempty(Ss)
            Z   = [ones(numel(Ss),1), Ss];
            Bj  = Z \ ys;                               % [const; beta_slope]
            yhat0 = [1, sj_new] * Bj;                   % 테스트 입력: t+H의 slope
        else
            Z   = ones(numel(y1),1);
            Bj  = Z \ y1;                               % 상수만
            yhat0 = [1] * Bj;
        end
        err_0(i,j)  = y_true - yhat0;
        B{i,j}      = Bj;

        % ---------- λ 선택: 확장창(block, forward-chaining) CV ----------
        % (1) Ridge on X
        [lambda1, lamgrid1, cvm1, cvsd1, lambda1_1se] = select_lambda_expanding_cv(X1, y1, base_opts, K);
        fit1    = glmnet(X1, y1, 'gaussian', setfield(base_opts,'lambda',lambda1));
        yhat1   = glmnetPredict(fit1, x_new, lambda1, 'response');
        err_1(i,j)  = y_true - yhat1;
        Beta_1{i,j} = glmnetCoef(fit1, lambda1);
        LAMPATH_1{i,j} = lamgrid1;  CVM_1{i,j} = cvm1;  CVSD_1{i,j} = cvsd1;
        LAM_1(i,j) = lambda1;       LAM1SE_1(i,j) = lambda1_1se;

        % (2) CS-residual + Ridge(X only)
        if ~isempty(Ss)
            resY2   = ys - [ones(numel(Ss),1), Ss] * Bj;   % 동시시점 CS 잔차
            [lambda2, lamgrid2, cvm2, cvsd2, lambda2_1se] = select_lambda_expanding_cv(Xs, resY2, base_opts, K);
            fit2    = glmnet(Xs, resY2, 'gaussian', setfield(base_opts,'lambda',lambda2));
            add2    = glmnetPredict(fit2, x_new, lambda2, 'response');  % 입력: X_{t+H}
            yhat2   = [1, sj_new]*Bj + add2;
            err_2(i,j)  = y_true - yhat2;
            Beta_2{i,j} = glmnetCoef(fit2, lambda2);
            LAMPATH_2{i,j} = lamgrid2;  CVM_2{i,j} = cvm2;  CVSD_2{i,j} = cvsd2;
            LAM_2(i,j) = lambda2;       LAM1SE_2(i,j) = lambda2_1se;
        else
            yhat2       = yhat1;
            err_2(i,j)  = y_true - yhat2;
            Beta_2{i,j} = [];
            LAMPATH_2{i,j} = []; CVM_2{i,j} = []; CVSD_2{i,j} = [];
            LAM_2(i,j) = NaN; LAM1SE_2(i,j) = NaN;
        end

        % (3) Ridge with penalized slope : [S_{s,j}, X_s]
        if ~isempty(Ss)
            X3      = [Ss, Xs];
            opts3   = base_opts; opts3.penalty_factor = ones(1, size(X3,2));
            [lambda3, lamgrid3, cvm3, cvsd3, lambda3_1se] = select_lambda_expanding_cv(X3, ys, opts3, K);
            fit3    = glmnet(X3, ys, 'gaussian', setfield(opts3,'lambda',lambda3));
            yhat3   = glmnetPredict(fit3, [sj_new, x_new], lambda3, 'response'); % 입력: t+H
            err_3(i,j)  = y_true - yhat3;
            Beta_3{i,j} = glmnetCoef(fit3, lambda3);
            LAMPATH_3{i,j} = lamgrid3;  CVM_3{i,j} = cvm3;  CVSD_3{i,j} = cvsd3;
            LAM_3(i,j) = lambda3;       LAM1SE_3(i,j) = lambda3_1se;
        else
            yhat3       = yhat1;
            err_3(i,j)  = y_true - yhat3;
            Beta_3{i,j} = [];
            LAMPATH_3{i,j} = []; CVM_3{i,j} = []; CVSD_3{i,j} = [];
            LAM_3(i,j) = NaN; LAM1SE_3(i,j) = NaN;
        end

        % (4) Ridge with *unpenalized* slope
        if ~isempty(Ss)
            X4      = [Ss, Xs];
            opts4   = base_opts; opts4.penalty_factor = [0, ones(1,size(Xs,2))];
            [lambda4, lamgrid4, cvm4, cvsd4, lambda4_1se] = select_lambda_expanding_cv(X4, ys, opts4, K);
            fit4    = glmnet(X4, ys, 'gaussian', setfield(opts4,'lambda',lambda4));
            yhat4   = glmnetPredict(fit4, [sj_new, x_new], lambda4, 'response'); % 입력: t+H
            err_4(i,j)  = y_true - yhat4;
            Beta_4{i,j} = glmnetCoef(fit4, lambda4);
            LAMPATH_4{i,j} = lamgrid4;  CVM_4{i,j} = cvm4;  CVSD_4{i,j} = cvsd4;
            LAM_4(i,j) = lambda4;       LAM1SE_4(i,j) = lambda4_1se;
        else
            yhat4       = yhat1;
            err_4(i,j)  = y_true - yhat4;
            Beta_4{i,j} = [];
            LAMPATH_4{i,j} = []; CVM_4{i,j} = []; CVSD_4{i,j} = [];
            LAM_4(i,j) = NaN; LAM1SE_4(i,j) = NaN;
        end

        % (5) slope-only (Ridge)
        if ~isempty(Ss)
            X5      = Ss; y5 = ys;
            opts5   = base_opts; opts5.penalty_factor = 1;
            [lambda5, lamgrid5, cvm5, cvsd5, lambda5_1se] = select_lambda_expanding_cv(X5, y5, opts5, K);
            fit5    = glmnet(X5, y5, 'gaussian', setfield(opts5,'lambda',lambda5));
            yhat5   = glmnetPredict(fit5, sj_new, lambda5, 'response');           % 입력: t+H
            err_5(i,j)  = y_true - yhat5;
            Beta_5{i,j} = glmnetCoef(fit5, lambda5);
            LAMPATH_5{i,j} = lamgrid5;  CVM_5{i,j} = cvm5;  CVSD_5{i,j} = cvsd5;
            LAM_5(i,j) = lambda5;       LAM1SE_5(i,j) = lambda5_1se;
        else
            yhat5       = yhat0;
            err_5(i,j)  = y_true - yhat5;
            Beta_5{i,j} = [];
            LAMPATH_5{i,j} = []; CVM_5{i,j} = []; CVSD_5{i,j} = [];
            LAM_5(i,j) = NaN; LAM1SE_5(i,j) = NaN;
        end

        if any(isnan([x_new, sj_new])) || isnan(y_true)
            warning('NaN in test (t+H) at Time=%d (j=%d).', Time(t+H), j);
        end
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
assignin('base','PREDICTORS', X_VAR_NAMES);
assignin('base','Beta_1',Beta_1); assignin('base','Beta_2',Beta_2);
assignin('base','Beta_3',Beta_3); assignin('base','Beta_4',Beta_4); assignin('base','Beta_5',Beta_5);
assignin('base','LAMPATH_1',LAMPATH_1); assignin('base','CVM_1',CVM_1); assignin('base','CVSD_1',CVSD_1);
assignin('base','LAMPATH_2',LAMPATH_2); assignin('base','CVM_2',CVM_2); assignin('base','CVSD_2',CVSD_2);
assignin('base','LAMPATH_3',LAMPATH_3); assignin('base','CVM_3',CVM_3); assignin('base','CVSD_3',CVSD_3);
assignin('base','LAMPATH_4',LAMPATH_4); assignin('base','CVM_4',CVM_4); assignin('base','CVSD_4',CVSD_4);
assignin('base','LAMPATH_5',LAMPATH_5); assignin('base','CVM_5',CVM_5); assignin('base','CVSD_5',CVSD_5);
assignin('base','LAM_1',LAM_1); assignin('base','LAM_2',LAM_2); assignin('base','LAM_3',LAM_3);
assignin('base','LAM_4',LAM_4); assignin('base','LAM_5',LAM_5);
assignin('base','LAM1SE_1',LAM1SE_1); assignin('base','LAM1SE_2',LAM1SE_2);
assignin('base','LAM1SE_3',LAM1SE_3); assignin('base','LAM1SE_4',LAM1SE_4); assignin('base','LAM1SE_5',LAM1SE_5);
assignin('base','ERR_BASE',err_0); assignin('base','ERR_1',err_1); assignin('base','ERR_2',err_2);
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

function [lambda_sel, lambda_grid, cvm, cvsd, lambda_1se] = select_lambda_expanding_cv(X, y, opts, K)
% 확장창(expanding window) 블록 CV(=forward-chaining)로 λ 선택.
% 1) n개 관측을 시간순으로 K개 블록으로 나눈 뒤
% 2) k=1..K-1: train = [1..edge(k+1)], validate = (edge(k+1)+1 .. edge(k+2))
% 3) fold별 MSE를 누적, per-λ로 평균(cvm)과 표준편차(cvsd) 집계
% 4) lambda_min = argmin(cvm); lambda_1se = 가장 큰 λ s.t. cvm(λ) <= cvm_min + cvsd_min
    n = size(X,1);
    assert(n >= 2, 'Too few observations for expanding-window CV.');
    K = max(2, min(K, n));
    edges = round(linspace(0, n, K+1));  % 0, ..., n

    % λ 그리드
    if isfield(opts,'lambda') && ~isempty(opts.lambda)
        lambda_grid = opts.lambda(:);
    else
        fit_master = glmnet(X, y, 'gaussian', rmfield_if(opts,'lambda'));
        lambda_grid = fit_master.lambda(:);
    end
    L = numel(lambda_grid);

    % 누적 통계
    sum_mse   = zeros(L,1);
    sumsq_mse = zeros(L,1);
    n_folds   = 0;

    for k = 1:(K-1)
        tr_hi = edges(k+1);        % train 마지막 인덱스
        va_lo = edges(k+1)+1;      % val 시작
        va_hi = edges(k+2);        % val 마지막

        if tr_hi < 5 || va_hi < va_lo
            continue;
        end

        Xtr = X(1:tr_hi,:);  ytr = y(1:tr_hi);
        Xva = X(va_lo:va_hi,:);  yva = y(va_lo:va_hi);

        fit  = glmnet(Xtr, ytr, 'gaussian', setfield(opts,'lambda',lambda_grid)); %#ok<SFLD>
        pred = glmnetPredict(fit, Xva, lambda_grid, 'response');   % |V|×L
        mse_k= mean((yva - pred).^2, 1)';                          % L×1

        sum_mse   = sum_mse   + mse_k;
        sumsq_mse = sumsq_mse + mse_k.^2;
        n_folds   = n_folds + 1;
    end

    % 평균/표준편차(표준오차는 fold 단위로 계산)
    cvm = sum_mse ./ n_folds;
    if n_folds > 1
        var_mse = (sumsq_mse - (sum_mse.^2)/n_folds) ./ (n_folds-1);
        cvsd = sqrt(var_mse) ./ sqrt(n_folds);
    else
        cvsd = zeros(L,1);
    end

    % lambda_min & 1SE 규칙
    [~, ix_min] = min(cvm);
    lambda_sel  = lambda_grid(ix_min);
    thresh      = cvm(ix_min) + cvsd(ix_min);
    % glmnet lambda는 보통 내림차순(큰→작). 임계 이내 중 "가장 큰" λ를 택함.
    ix_1se = find(cvm <= thresh, 1, 'first');
    if isempty(ix_1se), ix_1se = ix_min; end
    lambda_1se = lambda_grid(ix_1se);
end

function S = rmfield_if(S, fname)
    if isfield(S, fname), S = rmfield(S, fname); end
end