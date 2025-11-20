%% ====================== MAIN MLP (Intel MATLAB 2024b, TCV) ===================
% Logic matches 'main_tcv.m' (Ridge) exactly including Grid Search CV.
% Method: CS (Fixed OLS) + Residual MLP (via b2bLayer)
% Validation: K-Block Expanding Window Cross-Validation per step
% Warning: Computationally Expensive!
% ==============================================================================
clear; clc; close all;
dbstop if error
rng(1); % Matches main_tcv.m seed
addpath(genpath(cd));

%% ------------------------------ Experiment Setting ---------------------------
DATA_FILE    = "dataset.csv";
CACHE_FILE   = "main_mlp_tcv_cache.mat";
isreload     = true;

% Period / Embargo
H             = 12;              
PERIOD_START  = 199109;
PERIOD_END    = 202312;
BURN_START    = 199109;
BURN_END      = 201009;

GAP_MONTHS    = 12;  % Embargo

% Target Maturities
MATURITIES = ["xr_2","xr_5","xr_7","xr_10"];

% Predictors
spec_str   = "CP";

% MLP Hyperparameters
hyper.numNeurons      = 64;
hyper.numHiddenLayers = 2;
hyper.maxEpochs       = 200;  % Epochs per training
hyper.miniBatch       = 64;
hyper.learnRate       = 1e-3;
hyper.patience        = 10; 

% Grid Search Candidates (Weight Decay / L2 Penalty)
WD_GRID = [0, 1e-4, 1e-2]; 
K_BLOCKS = 5;  % Number of CV blocks

useGPU = (exist('canUseGPU','file') && canUseGPU());
fprintf('[INFO] GPU Mode: %s | Grid Size: %d | K-Blocks: %d\n', ...
    string(useGPU), numel(WD_GRID), K_BLOCKS);

%% ------------------------------- LOAD (Merged) ------------------------------
if isreload || ~isfile(CACHE_FILE)
    opts = detectImportOptions(DATA_FILE, 'TextType','string'); try, opts.VariableNamingRule='preserve'; catch, end
    T_raw = readtable(DATA_FILE, opts);

    % Time Processing
    assert(ismember("Time", string(T_raw.Properties.VariableNames)), "Time column required.");
    tcol = T_raw.("Time");
    if iscellstr(tcol) || isstring(tcol)
        s = string(tcol);
        y = str2double(extractBefore(s,5)); m = str2double(extractAfter(s,4));
        T_raw.("Time") = double(y*100 + m);
    elseif isnumeric(tcol)
        T_raw.("Time") = double(tcol(:));
    end
    T_raw = sortrows(T_raw,'Time');
    save(CACHE_FILE,'T_raw','-v7.3');
else
    S0 = load(CACHE_FILE,'T_raw'); T_raw = S0.T_raw;
end

% Filter Period
T = T_raw(T_raw.Time>=PERIOD_START & T_raw.Time<=PERIOD_END, :);

% Identify Y and Slope
Y_cols = MATURITIES(ismember(MATURITIES, string(T.Properties.VariableNames)));
slope_guess = regexprep(cellstr(Y_cols(:)), '^xr_', 's_');
assert(all(ismember(slope_guess, T.Properties.VariableNames)), 'Missing slope columns');

Ytbl = T(:, ["Time", Y_cols]);
Stbl = T(:, [{'Time'}, slope_guess(:)']);

Time = Ytbl.Time;
Y0   = tbl2mat_no_time(Ytbl);                 
S    = tbl2mat_no_time(Stbl);                 
[TN, J] = size(Y0);

Xtbl = build_X_from_spec_merged(T, spec_str, [string(Y_cols), string(slope_guess')]);
X    = tbl2mat_no_time(Xtbl);                 
p    = size(X,2);

X_VAR_NAMES = string(setdiff(Xtbl.Properties.VariableNames, {'Time'}, 'stable'));
fprintf('[INFO] Sample %d~%d (T=%d) | J=%d | p=%d\n', min(Time), max(Time), TN, J, p);

idx_burn_end = find(Time >= BURN_END, 1, 'first');
nOOS = TN - idx_burn_end - H; assert(nOOS>0, 'Not enough OOS.');

%% --------------------------- PREALLOC ---------------------------------------
err_0    = zeros(nOOS, J); 
err_mlp  = zeros(nOOS, J);
MLP_YHAT = nan(nOOS, J);

% Log selected hyperparams
BEST_WD  = nan(nOOS, 1); 

% [수정됨] Layer Names (%02d 포맷 적용)
outNames = arrayfun(@(x) sprintf('sum%02d', str2double(extractAfter(x,"xr_"))), Y_cols, 'UniformOutput', false);
mats_num = str2double(extractAfter(Y_cols, "xr_"));

% Acceleration
if exist('dlaccelerate','file')
    modelGradients_acc = dlaccelerate(@modelGradients);
    jointMSE_acc       = dlaccelerate(@jointMSE);
else
    modelGradients_acc = @modelGradients;
    jointMSE_acc       = @jointMSE;
end

%% ----------------------------- MAIN LOOP (TCV) ------------------------------
global_tic = tic;
fprintf('Starting TCV Loop (%d steps). This will take time...\n', nOOS);

for i = 1:nOOS
    t  = idx_burn_end + i - 1; % End of Training History

    % 1. Test Inputs (t+H)
    x_new = X(t+H, :);
    s_new = S(t+H, :);
    y_new = Y0(t+H, :);

    % 2. Available History (1:t)
    X_hist = X(1:t, :); Y_hist = Y0(1:t, :); S_hist = S(1:t, :);
    
    % Remove NaNs
    valid_rows = all(~isnan(X_hist),2) & all(~isnan(Y_hist),2) & all(~isnan(S_hist),2);
    X_hist = X_hist(valid_rows,:); Y_hist = Y_hist(valid_rows,:); S_hist = S_hist(valid_rows,:);
    
    % 3. Hyperparameter Selection (K-Block Expanding CV)
    % This matches Ridge's 'select_lambda_expanding_cv' logic
    [best_wd, cv_stats] = select_hyperparam_expanding_cv(...
        X_hist, Y_hist, S_hist, ...
        WD_GRID, K_BLOCKS, GAP_MONTHS, ...
        mats_num, hyper, useGPU, modelGradients_acc, jointMSE_acc, outNames);
    
    BEST_WD(i) = best_wd;
    
    % 4. Final Training (Refit on Full History 1:t with Best WD)
    % We use last 20% as Validation for Early Stopping
    n_hist  = size(X_hist, 1);
    n_val   = max(1, floor(0.2 * n_hist));
    n_train = n_hist - n_val;
    
    % Apply Gap for Final Train
    n_train_gap = max(1, n_train - GAP_MONTHS);
    
    X_tr = X_hist(1:n_train_gap, :); Y_tr = Y_hist(1:n_train_gap, :); S_tr = S_hist(1:n_train_gap, :);
    X_va = X_hist(n_train+1:end, :); Y_va = Y_hist(n_train+1:end, :); S_va = S_hist(n_train+1:end, :);
    
    % CS Baseline (OLS)
    CS_coefs = zeros(J, 2, 'single');
    for j = 1:J
        Z  = [ones(size(S_tr,1),1), S_tr(:,j)];
        if rank(Z) < 2, Bj=[0;0]; else, Bj=Z\Y_tr(:,j); end
        CS_coefs(j,:) = Bj';
        
        yhat0 = [1, s_new(j)] * Bj;
        err_0(i,j) = y_new(j) - yhat0;
    end
    
    % Prepare MLP Inputs
    X_net_tr = [S_tr, X_tr];
    X_net_va = [S_va, X_va];
    X_net_new= [s_new, x_new];
    
    % Set Best Hyperparam
    current_hyper = hyper;
    current_hyper.L2Regularization = best_wd;
    current_hyper.inputSize = size(X_net_tr, 2);
    current_hyper.outputSize = J;
    
    % Train Final Model
    [bestNet, ~] = train_mlp_single(X_net_tr, Y_tr, X_net_va, Y_va, ...
        mats_num, current_hyper, CS_coefs, useGPU, modelGradients_acc, jointMSE_acc, outNames);
    
    % 5. Predict
    dlX_new = dlarray(single(X_net_new'), 'CB');
    if useGPU, dlX_new = gpuArray(dlX_new); end
    
    for j = 1:J
        pred = extractdata(gather(forward(bestNet, dlX_new, 'Outputs', outNames{j})));
        MLP_YHAT(i,j) = pred;
        err_mlp(i,j)  = y_new(j) - MLP_YHAT(i,j);
    end
    
    if mod(i, 10)==0
        fprintf('Step %d/%d | Best WD: %.1e | Elapsed: %.1f s\n', ...
            i, nOOS, best_wd, toc(global_tic)); 
    end
end

%% ----------------------------- REPORT ---------------------------------------
R2 = @(e) 1 - sum(e.^2,1)./sum(err_0.^2,1);
R2_tbl = array2table([R2(err_mlp)], ...
    'VariableNames', cellstr(Y_cols), 'RowNames', {'TCV-MLP'});
disp(R2_tbl);

assignin('base','PREDICTORS', X_VAR_NAMES);
assignin('base','ERR_BASE',err_0);
assignin('base','ERR_MLP',err_mlp);
assignin('base','MLP_YHAT',MLP_YHAT);
assignin('base','BEST_WD',BEST_WD);

%% ============================ HELPERS =======================================
function [best_wd, stats] = select_hyperparam_expanding_cv(X, Y, S, grid, K, gap, mats, hyper, gpu, gradFn, lossFn, names)
% Mimics select_lambda_expanding_cv
    n = size(X,1);
    edges = round(linspace(0, n, K+1));
    
    L = numel(grid);
    mean_mse = zeros(L, 1);
    J = size(Y, 2);
    
    for g = 1:L
        wd = grid(g);
        cur_hyper = hyper;
        cur_hyper.L2Regularization = wd;
        
        fold_mses = [];
        
        for k = 1:(K-1)
            idx_split = edges(k+1);
            idx_val_end = edges(k+2);
            
            if idx_split < 20, continue; end
            
            % Apply Gap to Train
            n_train_gap = max(1, idx_split - gap);
            
            X_tr = X(1:n_train_gap, :); Y_tr = Y(1:n_train_gap, :); S_tr = S(1:n_train_gap, :);
            X_va = X(idx_split+1:idx_val_end, :); Y_va = Y(idx_split+1:idx_val_end, :); S_va = S(idx_split+1:idx_val_end, :);
            
            if isempty(X_va), continue; end
            
            % Fit CS
            CS_c = zeros(J, 2, 'single');
            for j=1:J
                des = [ones(size(S_tr,1),1), S_tr(:,j)];
                if rank(des) < 2, b=[0;0]; else, b=des\Y_tr(:,j); end
                CS_c(j,:) = b';
            end
            
            % Prep Nets
            Xt = [S_tr, X_tr]; Xv = [S_va, X_va];
            cur_hyper.inputSize = size(Xt, 2);
            cur_hyper.outputSize = J;
            
            % Train
            [~, bestLoss] = train_mlp_single(Xt, Y_tr, Xv, Y_va, mats, cur_hyper, CS_c, gpu, gradFn, lossFn, names);
            fold_mses = [fold_mses; bestLoss];
        end
        
        if isempty(fold_mses)
            mean_mse(g) = inf;
        else
            mean_mse(g) = mean(fold_mses);
        end
    end
    
    [~, idx_min] = min(mean_mse);
    best_wd = grid(idx_min);
    stats.mean_mse = mean_mse;
end

function [bestNet, bestValLoss] = train_mlp_single(Xtr, Ytr, Xva, Yva, mats, hyper, CS, gpu, gradFn, lossFn, names)
    dlnet = b2bLayer(mats, hyper, CS, false);
    if gpu, dlnet = dlupdate(@gpuArray, dlnet); end
    
    dlX_tr = dlarray(single(Xtr'), 'CB'); dlY_tr = dlarray(single(Ytr'), 'CB');
    dlX_va = dlarray(single(Xva'), 'CB'); dlY_va = dlarray(single(Yva'), 'CB');
    if gpu, [dlX_tr, dlY_tr, dlX_va, dlY_va] = deal(gpuArray(dlX_tr), gpuArray(dlY_tr), gpuArray(dlX_va), gpuArray(dlY_va)); end
    
    trailingAvg = []; trailingAvgSq = [];
    bestValLoss = inf; bestNet = dlnet;
    patience = 0; iter = 0; nTr = size(Xtr, 1);
    
    for epoch = 1:hyper.maxEpochs
        idx = randperm(nTr);
        for k = 1:hyper.miniBatch:nTr
            iter = iter + 1;
            b = idx(k : min(k + hyper.miniBatch - 1, nTr));
            [loss, grads] = dlfeval(gradFn, dlnet, dlX_tr(:,b), dlY_tr(:,b), names);
            
            % Apply L2 Regularization Manually
            if isfield(hyper, 'L2Regularization') && hyper.L2Regularization > 0
                lambda = hyper.L2Regularization;
                for pIdx = 1:numel(grads.Value)
                    grads.Value{pIdx} = grads.Value{pIdx} + lambda * dlnet.Learnables.Value{pIdx};
                end
            end
            
            [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, grads, trailingAvg, trailingAvgSq, iter, hyper.learnRate);
        end
        
        vLoss = extractdata(gather(dlfeval(lossFn, dlnet, dlX_va, dlY_va, names)));
        if vLoss < bestValLoss
            bestValLoss = vLoss; bestNet = dlnet; patience = 0;
        else
            patience = patience + 1;
        end
        if patience >= hyper.patience, break; end
    end
end

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
    vn   = string(T.Properties.VariableNames);
    reserved = lower(["Time", string(reserved_list(:)')]);
    raw = regexp(char(spec_str), '\s*[+,]\s*', 'split');
    want = unique(string(strtrim(raw)),"stable");
    selected = strings(0,1);
    for w = want(:)'
        if w==""; continue; end
        idx = find(strcmpi(vn, w), 1, 'first');
        if ~isempty(idx)
            cand = vn(idx);
            if any(strcmpi(reserved, cand)), continue; end
            selected(end+1,1) = cand;
        end
    end
    Xtbl = T(:, [{'Time'}, cellstr(selected)']);
end

function [L, G] = modelGradients(dlnet, dlX, dlY, outNames)
    Jloc = size(dlY,1); acc  = dlarray(0, 'CB');
    for k = 1:Jloc, yk = forward(dlnet, dlX, 'Outputs', outNames{k}); d = yk - dlY(k,:); acc = acc + mean(d.^2, 'all'); end
    L = acc / Jloc; G = dlgradient(L, dlnet.Learnables);
end

function L = jointMSE(dlnet, dlX, dlY, outNames)
    Jloc = size(dlY,1); acc  = dlarray(0, 'CB');
    for k = 1:Jloc, yk = forward(dlnet, dlX, 'Outputs', outNames{k}); d = yk - dlY(k,:); acc = acc + mean(d.^2, 'all'); end
    L = acc / Jloc;
end