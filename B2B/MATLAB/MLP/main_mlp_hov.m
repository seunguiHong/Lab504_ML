%% ====================== MAIN MLP (Intel MATLAB 2024b, hov) ===================
% Two Experiments:
% 1. CS-Residual MLP: y = (alpha + beta*s) + MLP(s, X)
% 2. Simple MLP:      y = MLP(s, X)
% Validation: Chronological Holdout + L2 Grid Search
% ==============================================================================
clear; clc; close all;
dbstop if error
rng(0);
addpath(genpath(cd));

%% ------------------------------ Experiment Setting ---------------------------
DATA_FILE    = "dataset.csv";
CACHE_FILE   = "main_mlp_hov.mat";
isreload     = true;

H             = 12;              
PERIOD_START  = 197108;
PERIOD_END    = 202312;
BURN_START    = 197108;
BURN_END      = 197201;

MATURITIES = ["xr_2","xr_3","xr_4","xr_5","xr_6","xr_7","xr_8","xr_9","xr_10"];
spec_str   = "F1, F3, F4, F8, F1^3";
VAL_FRAC   = 0.30; 

% MLP Hyperparameters
hyper.numNeurons      = 64;
hyper.numHiddenLayers = 2;
hyper.maxEpochs       = 200;
hyper.miniBatch       = 64;
hyper.learnRate       = 1e-3;
hyper.patience        = 10; 

% Grid Search Candidates
WD_GRID = [0, 1e-4, 1e-2];

useGPU = (exist('canUseGPU','file') && canUseGPU());
fprintf('[INFO] GPU Mode: %s | Grid: %s\n', string(useGPU), mat2str(WD_GRID));

%% ------------------------------- LOAD ---------------------------------------
if isreload || ~isfile(CACHE_FILE)
    opts = detectImportOptions(DATA_FILE, 'TextType','string'); try, opts.VariableNamingRule='preserve'; catch, end
    T = readtable(DATA_FILE, opts);
    assert(ismember("Time", string(T.Properties.VariableNames)), "Time column required.");
    tcol = T.("Time");
    if iscellstr(tcol) || isstring(tcol)
        s = string(tcol); y = str2double(extractBefore(s,5)); m = str2double(extractAfter(s,4));
        T.("Time") = double(y*100 + m);
    elseif isnumeric(tcol), T.("Time") = double(tcol(:)); end
    T = sortrows(T,'Time');
    T = T(T.Time>=PERIOD_START & T.Time<=PERIOD_END, :);
    Y_cols = MATURITIES(ismember(MATURITIES, string(T.Properties.VariableNames)));
    slope_guess = regexprep(cellstr(Y_cols(:)), '^xr_', 's_');
    assert(all(ismember(slope_guess, T.Properties.VariableNames)), 'Missing slope columns');
    save(CACHE_FILE, 'T','Y_cols','slope_guess','PERIOD_START','PERIOD_END','-v7.3');
else
    S_load = load(CACHE_FILE, 'T','Y_cols','slope_guess','PERIOD_START','PERIOD_END');
    T = S_load.T; Y_cols = S_load.Y_cols; slope_guess = S_load.slope_guess;
end
assignin('base','DATASET',T);

%% ------------------------------- BUILD ---------------------------------------
Ytbl = T(:, ["Time", Y_cols]); Time = Ytbl.Time; Y0 = tbl2mat_no_time(Ytbl);                 
[TN, J] = size(Y0);
Stbl = T(:, [{'Time'}, slope_guess(:)']); S = tbl2mat_no_time(Stbl);                 
Xtbl = build_X_from_spec_merged(T, spec_str, [string(Y_cols), string(slope_guess')]);
X = tbl2mat_no_time(Xtbl); p = size(X,2);
X_VAR_NAMES = string(setdiff(Xtbl.Properties.VariableNames, {'Time'}, 'stable'));

idx_burn_end = find(Time >= BURN_END, 1, 'first');
nOOS = TN - idx_burn_end - H; assert(nOOS>0, 'Not enough OOS.');

%% --------------------------- PREALLOC ---------------------------------------
% Baseline
err_0 = zeros(nOOS, J); CS_YHAT = nan(nOOS, J);

% Exp 1: CS-Residual MLP
err_csmlp = zeros(nOOS, J); CSMLP_YHAT = nan(nOOS, J); BestWD_CSMLP = nan(nOOS,1);

% Exp 2: Simple MLP
err_sim   = zeros(nOOS, J); SIM_YHAT   = nan(nOOS, J); BestWD_SIM   = nan(nOOS,1);

outNames = arrayfun(@(x) sprintf('sum%02d', str2double(extractAfter(x,"xr_"))), Y_cols, 'UniformOutput', false);
mats_num = str2double(extractAfter(Y_cols, "xr_")); 

if exist('dlaccelerate','file')
    modelGradients_acc = dlaccelerate(@modelGradients);
    jointMSE_acc       = dlaccelerate(@jointMSE);
else
    modelGradients_acc = @modelGradients;
    jointMSE_acc       = @jointMSE;
end

%% ----------------------------- MAIN LOOP ------------------------------------
tic
fprintf('Starting Loop (%d steps)...\n', nOOS);

for i = 1:nOOS
    t  = idx_burn_end + i - 1;

    % 1. Data Slicing
    x_new = X(t+H, :); s_new = S(t+H, :); y_new = Y0(t+H, :);
    X_full = X(1:t, :); Y_full = Y0(1:t, :); S_full = S(1:t, :);
    
    n_full  = size(X_full, 1);
    n_val   = max(1, floor(VAL_FRAC * n_full));
    n_train = n_full - n_val;
    
    X_tr = X_full(1:n_train, :); Y_tr = Y_full(1:n_train, :); S_tr = S_full(1:n_train, :);
    X_va = X_full(n_train+1:end, :); Y_va = Y_full(n_train+1:end, :); S_va = S_full(n_train+1:end, :);
    
    % 2. CS Baseline
    CS_coefs = zeros(J, 2, 'single');
    for j = 1:J
        Z  = [ones(size(S_tr,1),1), S_tr(:,j)];
        if rank(Z) < 2, Bj=[0;0]; else, Bj=Z\Y_tr(:,j); end
        CS_coefs(j,:) = Bj';
        yhat0 = [1, s_new(j)] * Bj;
        err_0(i,j) = y_new(j) - yhat0;
        CS_YHAT(i,j) = yhat0;
    end
    
    % Inputs for MLP: [Slope, Predictors]
    X_net_tr = [S_tr, X_tr]; X_net_va = [S_va, X_va]; X_net_new = [s_new, x_new];
    hyper.inputSize = size(X_net_tr, 2);
    hyper.outputSize = J;
    
    % =====================================================================
    % Experiment 1: CS-Residual MLP (Grid Search)
    % =====================================================================
    best_loss_cs = inf; best_net_cs = []; best_wd_cs = NaN;
    
    for wd = WD_GRID
        h_cand = hyper; h_cand.L2Regularization = wd;
        % Model Type: 'cs_resid' uses b2bLayer
        [net, v_loss] = train_mlp_generic(X_net_tr, Y_tr, X_net_va, Y_va, ...
            'cs_resid', mats_num, h_cand, CS_coefs, ...
            useGPU, modelGradients_acc, jointMSE_acc, outNames);
        
        if v_loss < best_loss_cs
            best_loss_cs = v_loss; best_net_cs = net; best_wd_cs = wd;
        end
    end
    BestWD_CSMLP(i) = best_wd_cs;
    
    % Predict
    dlX_new = dlarray(single(X_net_new'), 'CB'); if useGPU, dlX_new = gpuArray(dlX_new); end
    for j = 1:J
        pred = extractdata(gather(forward(best_net_cs, dlX_new, 'Outputs', outNames{j})));
        CSMLP_YHAT(i,j) = pred;
        err_csmlp(i,j)  = y_new(j) - pred;
    end
    
    % =====================================================================
    % Experiment 2: Simple MLP (Grid Search)
    % =====================================================================
    best_loss_sim = inf; best_net_sim = []; best_wd_sim = NaN;
    
    for wd = WD_GRID
        h_cand = hyper; h_cand.L2Regularization = wd;
        % Model Type: 'simple' uses standard MLP graph
        [net, v_loss] = train_mlp_generic(X_net_tr, Y_tr, X_net_va, Y_va, ...
            'simple', mats_num, h_cand, [], ...
            useGPU, modelGradients_acc, jointMSE_acc, outNames);
        
        if v_loss < best_loss_sim
            best_loss_sim = v_loss; best_net_sim = net; best_wd_sim = wd;
        end
    end
    BestWD_SIM(i) = best_wd_sim;
    
    % Predict
    for j = 1:J
        pred = extractdata(gather(forward(best_net_sim, dlX_new, 'Outputs', outNames{j})));
        SIM_YHAT(i,j) = pred;
        err_sim(i,j)  = y_new(j) - pred;
    end
    
    if mod(i, 10)==0
        fprintf('Step %d/%d | CS-MLP WD=%.1g | Sim-MLP WD=%.1g\n', i, nOOS, best_wd_cs, best_wd_sim); 
    end
end
toc

%% ----------------------------- REPORT ---------------------------------------
R2 = @(e) 1 - sum(e.^2,1)./sum(err_0.^2,1);
R2_tbl = array2table([R2(err_csmlp); R2(err_sim)], ...
    'VariableNames', cellstr(Y_cols), ...
    'RowNames', {'CS-Residual MLP', 'Simple MLP'});
disp(R2_tbl);

assignin('base','ERR_CSMLP',err_csmlp);
assignin('base','ERR_SIM',err_sim);
assignin('base','CSMLP_YHAT',CSMLP_YHAT);
assignin('base','SIM_YHAT',SIM_YHAT);

%% ============================ HELPERS =======================================
function [bestNet, bestValLoss] = train_mlp_generic(Xtr, Ytr, Xva, Yva, type, mats, hyper, CS, gpu, gradFn, lossFn, names)
    % Build Graph based on Type
    if strcmp(type, 'cs_resid')
        dlnet = b2bLayer(mats, hyper, CS, false);
    else
        % Simple MLP Construction
        % Input -> FC -> ELU -> FC -> ELU -> Heads
        inSz = hyper.inputSize; nH = hyper.numNeurons;
        lgraph = layerGraph(featureInputLayer(inSz, 'Name', 'input'));
        
        % Trunk
        lgraph = addLayers(lgraph, [
            fullyConnectedLayer(nH, 'Name', 'fc1', 'WeightsInitializer','glorot')
            eluLayer('Name', 'elu1')
            fullyConnectedLayer(nH, 'Name', 'fc2', 'WeightsInitializer','glorot')
            eluLayer('Name', 'elu2')
        ]);
        lgraph = connectLayers(lgraph, 'input', 'fc1');
        
        % Heads (to match outNames logic)
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
    
    trailingAvg = []; trailingAvgSq = []; bestValLoss = inf; bestNet = dlnet;
    patience = 0; iter = 0; nTr = size(Xtr, 1);
    
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
    for w = want(:)', if w=="", continue; end; idx = find(strcmpi(vn, w), 1, 'first');
    if ~isempty(idx), cand = vn(idx); if any(strcmpi(reserved, cand)), continue; end; selected(end+1,1) = cand; end; end
    Xtbl = T(:, [{'Time'}, cellstr(selected)']);
end

function [L, G] = modelGradients(dlnet, dlX, dlY, outNames)
    Jloc = size(dlY,1); acc = dlarray(0, 'CB');
    for k = 1:Jloc, yk = forward(dlnet, dlX, 'Outputs', outNames{k}); d = yk - dlY(k,:); acc = acc + mean(d.^2, 'all'); end
    L = acc / Jloc; G = dlgradient(L, dlnet.Learnables);
end

function L = jointMSE(dlnet, dlX, dlY, outNames)
    Jloc = size(dlY,1); acc = dlarray(0, 'CB');
    for k = 1:Jloc, yk = forward(dlnet, dlX, 'Outputs', outNames{k}); d = yk - dlY(k,:); acc = acc + mean(d.^2, 'all'); end
    L = acc / Jloc;
end