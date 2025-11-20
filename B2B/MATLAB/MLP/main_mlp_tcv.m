%% ====================== MAIN MLP (Intel MATLAB 2024b, TCV) ===================
% Two Experiments: CS-Residual MLP & Simple MLP
% Validation: K-Block Expanding Window Cross-Validation
% ==============================================================================
clear; clc; close all;
dbstop if error
rng(1);
addpath(genpath(cd));

%% ------------------------------ Experiment Setting ---------------------------
DATA_FILE    = "dataset.csv";
CACHE_FILE   = "main_mlp_tcv_cache.mat";
isreload     = true;

H             = 12;              
PERIOD_START  = 199109;
PERIOD_END    = 202312;
BURN_START    = 199109;
BURN_END      = 201009;
GAP_MONTHS    = 12;

MATURITIES = ["xr_2","xr_5","xr_7","xr_10"];
spec_str   = "CP";

hyper.numNeurons      = 64;
hyper.numHiddenLayers = 2;
hyper.maxEpochs       = 200;
hyper.miniBatch       = 64;
hyper.learnRate       = 1e-3;
hyper.patience        = 10; 

WD_GRID = [0, 1e-4, 1e-2]; 
K_BLOCKS = 5;  

useGPU = (exist('canUseGPU','file') && canUseGPU());
fprintf('[INFO] GPU Mode: %s | Grid: %s\n', string(useGPU), mat2str(WD_GRID));

%% ------------------------------- LOAD ---------------------------------------
if isreload || ~isfile(CACHE_FILE)
    opts = detectImportOptions(DATA_FILE, 'TextType','string'); try, opts.VariableNamingRule='preserve'; catch, end
    T_raw = readtable(DATA_FILE, opts);
    assert(ismember("Time", string(T_raw.Properties.VariableNames)), "Time column required.");
    tcol = T_raw.("Time");
    if iscellstr(tcol) || isstring(tcol)
        s = string(tcol); y = str2double(extractBefore(s,5)); m = str2double(extractAfter(s,4));
        T_raw.("Time") = double(y*100 + m);
    elseif isnumeric(tcol), T_raw.("Time") = double(tcol(:)); end
    T_raw = sortrows(T_raw,'Time');
    save(CACHE_FILE,'T_raw','-v7.3');
else
    S0 = load(CACHE_FILE,'T_raw'); T_raw = S0.T_raw;
end
T = T_raw(T_raw.Time>=PERIOD_START & T_raw.Time<=PERIOD_END, :);
Y_cols = MATURITIES(ismember(MATURITIES, string(T.Properties.VariableNames)));
slope_guess = regexprep(cellstr(Y_cols(:)), '^xr_', 's_');
assert(all(ismember(slope_guess, T.Properties.VariableNames)), 'Missing slope columns');

Ytbl = T(:, ["Time", Y_cols]); Time = Ytbl.Time; Y0 = tbl2mat_no_time(Ytbl);                 
Stbl = T(:, [{'Time'}, slope_guess(:)']); S = tbl2mat_no_time(Stbl);                 
[TN, J] = size(Y0);
Xtbl = build_X_from_spec_merged(T, spec_str, [string(Y_cols), string(slope_guess')]);
X = tbl2mat_no_time(Xtbl); p = size(X,2);
X_VAR_NAMES = string(setdiff(Xtbl.Properties.VariableNames, {'Time'}, 'stable'));

idx_burn_end = find(Time >= BURN_END, 1, 'first');
nOOS = TN - idx_burn_end - H; assert(nOOS>0, 'Not enough OOS.');

%% --------------------------- PREALLOC ---------------------------------------
err_0     = zeros(nOOS, J);
err_csmlp = zeros(nOOS, J); CSMLP_YHAT = nan(nOOS, J);
err_sim   = zeros(nOOS, J); SIM_YHAT   = nan(nOOS, J);

outNames = arrayfun(@(x) sprintf('sum%02d', str2double(extractAfter(x,"xr_"))), Y_cols, 'UniformOutput', false);
mats_num = str2double(extractAfter(Y_cols, "xr_"));

if exist('dlaccelerate','file')
    modelGradients_acc = dlaccelerate(@modelGradients);
    jointMSE_acc       = dlaccelerate(@jointMSE);
else
    modelGradients_acc = @modelGradients;
    jointMSE_acc       = @jointMSE;
end

%% ----------------------------- MAIN LOOP (TCV) ------------------------------
global_tic = tic;
fprintf('Starting TCV Loop (%d steps)...\n', nOOS);

for i = 1:nOOS
    t  = idx_burn_end + i - 1;

    x_new = X(t+H, :); s_new = S(t+H, :); y_new = Y0(t+H, :);
    X_hist = X(1:t, :); Y_hist = Y0(1:t, :); S_hist = S(1:t, :);
    
    valid_rows = all(~isnan(X_hist),2) & all(~isnan(Y_hist),2) & all(~isnan(S_hist),2);
    X_hist = X_hist(valid_rows,:); Y_hist = Y_hist(valid_rows,:); S_hist = S_hist(valid_rows,:);
    
    % Final Fit Data Split (Last 20% for Early Stopping)
    n_hist = size(X_hist, 1); n_val = max(1, floor(0.2 * n_hist)); n_train = n_hist - n_val;
    n_train_gap = max(1, n_train - GAP_MONTHS);
    X_tr = X_hist(1:n_train_gap, :); Y_tr = Y_hist(1:n_train_gap, :); S_tr = S_hist(1:n_train_gap, :);
    X_va = X_hist(n_train+1:end, :); Y_va = Y_hist(n_train+1:end, :); S_va = S_hist(n_train+1:end, :);
    
    X_net_tr = [S_tr, X_tr]; X_net_va = [S_va, X_va]; X_net_new = [s_new, x_new];
    hyper.inputSize = size(X_net_tr, 2); hyper.outputSize = J;
    
    % CS Baseline
    CS_coefs = zeros(J, 2, 'single');
    for j = 1:J
        Z = [ones(size(S_tr,1),1), S_tr(:,j)];
        if rank(Z)<2, b=[0;0]; else, b=Z\Y_tr(:,j); end
        CS_coefs(j,:) = b';
        yhat0 = [1, s_new(j)] * b;
        err_0(i,j) = y_new(j) - yhat0;
    end

    % =====================================================================
    % Experiment 1: CS-Residual MLP (TCV)
    % =====================================================================
    % Select Hyperparam
    best_wd_cs = select_hyperparam_expanding_cv(X_hist, Y_hist, S_hist, WD_GRID, K_BLOCKS, GAP_MONTHS, ...
        'cs_resid', mats_num, hyper, useGPU, modelGradients_acc, jointMSE_acc, outNames);
    
    % Final Fit
    h_run = hyper; h_run.L2Regularization = best_wd_cs;
    bestNet_cs = train_mlp_generic(X_net_tr, Y_tr, X_net_va, Y_va, 'cs_resid', ...
        mats_num, h_run, CS_coefs, useGPU, modelGradients_acc, jointMSE_acc, outNames);
    
    % Predict
    dlX_new = dlarray(single(X_net_new'), 'CB'); if useGPU, dlX_new = gpuArray(dlX_new); end
    for j = 1:J
        CSMLP_YHAT(i,j) = extractdata(gather(forward(bestNet_cs, dlX_new, 'Outputs', outNames{j})));
        err_csmlp(i,j)  = y_new(j) - CSMLP_YHAT(i,j);
    end
    
    % =====================================================================
    % Experiment 2: Simple MLP (TCV)
    % =====================================================================
    best_wd_sim = select_hyperparam_expanding_cv(X_hist, Y_hist, S_hist, WD_GRID, K_BLOCKS, GAP_MONTHS, ...
        'simple', mats_num, hyper, useGPU, modelGradients_acc, jointMSE_acc, outNames);
        
    h_run = hyper; h_run.L2Regularization = best_wd_sim;
    bestNet_sim = train_mlp_generic(X_net_tr, Y_tr, X_net_va, Y_va, 'simple', ...
        mats_num, h_run, [], useGPU, modelGradients_acc, jointMSE_acc, outNames);
    
    for j = 1:J
        SIM_YHAT(i,j) = extractdata(gather(forward(bestNet_sim, dlX_new, 'Outputs', outNames{j})));
        err_sim(i,j)  = y_new(j) - SIM_YHAT(i,j);
    end
    
    if mod(i, 10)==0
        fprintf('Step %d/%d | CS-WD=%.1g | Sim-WD=%.1g | Elapsed: %.1f s\n', ...
            i, nOOS, best_wd_cs, best_wd_sim, toc(global_tic)); 
    end
end

%% ----------------------------- REPORT ---------------------------------------
R2 = @(e) 1 - sum(e.^2,1)./sum(err_0.^2,1);
R2_tbl = array2table([R2(err_csmlp); R2(err_sim)], ...
    'VariableNames', cellstr(Y_cols), 'RowNames', {'CS-Residual MLP', 'Simple MLP'});
disp(R2_tbl);

assignin('base','ERR_CSMLP',err_csmlp);
assignin('base','ERR_SIM',err_sim);
assignin('base','CSMLP_YHAT',CSMLP_YHAT);
assignin('base','SIM_YHAT',SIM_YHAT);

%% ============================ HELPERS =======================================
function best_wd = select_hyperparam_expanding_cv(X, Y, S, grid, K, gap, type, mats, hyper, gpu, gradFn, lossFn, names)
    n = size(X,1); edges = round(linspace(0, n, K+1)); L = numel(grid); mean_mse = zeros(L, 1); J = size(Y, 2);
    
    for g = 1:L
        wd = grid(g); cur_hyper = hyper; cur_hyper.L2Regularization = wd;
        fold_mses = [];
        
        for k = 1:(K-1)
            idx_split = edges(k+1); idx_val_end = edges(k+2);
            if idx_split < 20, continue; end
            n_train_gap = max(1, idx_split - gap);
            
            X_tr = X(1:n_train_gap, :); Y_tr = Y(1:n_train_gap, :); S_tr = S(1:n_train_gap, :);
            X_va = X(idx_split+1:idx_val_end, :); Y_va = Y(idx_split+1:idx_val_end, :); S_va = S(idx_split+1:idx_val_end, :);
            if isempty(X_va), continue; end
            
            CS_c = zeros(J, 2, 'single');
            if strcmp(type, 'cs_resid')
                for j=1:J
                    des=[ones(size(S_tr,1),1), S_tr(:,j)];
                    if rank(des)<2, b=[0;0]; else, b=des\Y_tr(:,j); end
                    CS_c(j,:)=b';
                end
            end
            
            Xt = [S_tr, X_tr]; Xv = [S_va, X_va];
            cur_hyper.inputSize = size(Xt, 2); cur_hyper.outputSize = J;
            
            [~, bestLoss] = train_mlp_generic(Xt, Y_tr, Xv, Y_va, type, mats, cur_hyper, CS_c, gpu, gradFn, lossFn, names);
            fold_mses = [fold_mses; bestLoss];
        end
        
        if isempty(fold_mses), mean_mse(g) = inf; else, mean_mse(g) = mean(fold_mses); end
    end
    [~, idx_min] = min(mean_mse); best_wd = grid(idx_min);
end

function [bestNet, bestValLoss] = train_mlp_generic(Xtr, Ytr, Xva, Yva, type, mats, hyper, CS, gpu, gradFn, lossFn, names)
    if strcmp(type, 'cs_resid')
        dlnet = b2bLayer(mats, hyper, CS, false);
    else
        inSz = hyper.inputSize; nH = hyper.numNeurons;
        lgraph = layerGraph(featureInputLayer(inSz, 'Name', 'input'));
        lgraph = addLayers(lgraph, [
            fullyConnectedLayer(nH, 'Name', 'fc1', 'WeightsInitializer','glorot'); eluLayer('Name', 'elu1')
            fullyConnectedLayer(nH, 'Name', 'fc2', 'WeightsInitializer','glorot'); eluLayer('Name', 'elu2')
        ]);
        lgraph = connectLayers(lgraph, 'input', 'fc1');
        for j = 1:hyper.outputSize
            hName = names{j};
            lgraph = addLayers(lgraph, fullyConnectedLayer(1, 'Name', hName, 'WeightsInitializer','glorot'));
            lgraph = connectLayers(lgraph, 'elu2', hName);
        end
        dlnet = dlnetwork(lgraph);
    end

    if gpu, dlnet = dlupdate(@gpuArray, dlnet); end
    dlX_tr = dlarray(single(Xtr'), 'CB'); dlY_tr = dlarray(single(Ytr'), 'CB');
    dlX_va = dlarray(single(Xva'), 'CB'); dlY_va = dlarray(single(Yva'), 'CB');
    if gpu, [dlX_tr, dlY_tr, dlX_va, dlY_va] = deal(gpuArray(dlX_tr), gpuArray(dlY_tr), gpuArray(dlX_va), gpuArray(dlY_va)); end
    
    trailingAvg = []; trailingAvgSq = []; bestValLoss = inf; bestNet = dlnet; patience = 0; iter = 0; nTr = size(Xtr, 1);
    for epoch = 1:hyper.maxEpochs
        idx = randperm(nTr);
        for k = 1:hyper.miniBatch:nTr
            iter = iter + 1; b = idx(k : min(k + hyper.miniBatch - 1, nTr));
            [loss, grads] = dlfeval(gradFn, dlnet, dlX_tr(:,b), dlY_tr(:,b), names);
            if isfield(hyper, 'L2Regularization') && hyper.L2Regularization > 0
                lam = hyper.L2Regularization;
                for pIdx = 1:numel(grads.Value), grads.Value{pIdx} = grads.Value{pIdx} + lam * dlnet.Learnables.Value{pIdx}; end
            end
            [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, grads, trailingAvg, trailingAvgSq, iter, hyper.learnRate);
        end
        vLoss = extractdata(gather(dlfeval(lossFn, dlnet, dlX_va, dlY_va, names)));
        if vLoss < bestValLoss, bestValLoss = vLoss; bestNet = dlnet; patience = 0; else, patience = patience + 1; end
        if patience >= hyper.patience, break; end
    end
end

function M = tbl2mat_no_time(T)
    try, T = removevars(T, 'Time'); catch, idx=find(strcmp('Time',T.Properties.VariableNames),1); T=T(:,setdiff(1:width(T),idx)); end
    M = double(table2array(T));
end

function Xtbl = build_X_from_spec_merged(T, spec_str, reserved_list)
    vn = string(T.Properties.VariableNames); reserved = lower(["Time", string(reserved_list(:)')]);
    raw = regexp(char(spec_str), '\s*[+,]\s*', 'split'); want = unique(string(strtrim(raw)),"stable");
    selected = strings(0,1);
    for w = want(:)', if w=="", continue; end; idx = find(strcmpi(vn