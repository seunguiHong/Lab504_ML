% ========================================================================
% Main: CS(fixed) + Shared Residual MLP (joint across maturities)
% (Seed-fixed version of the original)
%
% Purpose:
% - Step 1: Estimate a linear Campbell–Shiller (CS) regression separately
%           for each maturity: y_i = alpha_i + beta_i * slope_i.
% - Step 2: Build a multi-output MLP that learns *residuals* jointly across
%           maturities, with a shared hidden representation.
% - Step 3: Train the MLP with Adam on a fixed train/validation split,
%           keeping the CS coefficients fixed throughout.
% ========================================================================
clear; clc; close all;
addpath(genpath(cd));              % Add current folder and subfolders to path

% ------------------------------ Config -----------------------------------
VERBOSE      = 0;                  % b2bLayer internal verbosity switch
isreload     = 0;                  % 1: rebuild from CSV, 0: load from MAT cache
useGPUMode   = 'cpu';              % 'auto' | 'gpu' | 'cpu' (training device)
maturities   = [2, 3, 5, 7, 10];   % Bond maturities (years) used as targets
J            = numel(maturities);  % Number of maturities (output heads)

% Reproducibility seed for:
% - CS OLS estimation
% - Network initialization inside b2bLayer
% - Mini-batch shuffling (randperm)
RNG_SEED     = 42;

% Data directory and file names
dataDir = fullfile('data');
csvFile = fullfile(dataDir, 'dataset.csv');   % Raw CSV used when isreload=1
matFile = fullfile(dataDir, 'matdat.mat');    % Preprocessed MAT cache

% ------------------------- Load / Prepare Data ---------------------------
if isreload
    % Construct target variable names "xr_2", "xr_3", ..., to match CSV headers.
    xrNames = cellfun(@(x) ['xr_', num2str(x)], num2cell(maturities), 'UniformOutput', false);

    % Prepare() handles:
    % - Period selection
    % - Burn-in truncation
    % - Selection of forward-rate predictors
    out = prepare(csvFile, ...
        'Period',   [197108 202312], ...
        'Maturities', xrNames, ...
        'BurnEnd',  199001, ...
        'XInclude', ["fwd_2","fwd_3","fwd_4","fwd_5","fwd_6","fwd_7","fwd_8","fwd_9","fwd_10"]);

    % X: predictor matrix (slopes + additional covariates)
    % Y: stacked excess returns across maturities
    X = out.X; 
    Y = out.Y; 
    X_name = out.X_VAR_NAMES; %#ok<NASGU>   % Predictor names kept for inspection

    % Cache preprocessed data to speed up subsequent runs
    save(matFile, 'X','Y','X_name','maturities','-v7.3');
else
    % Load preprocessed dataset from MAT file
    L = load(matFile);
    X = L.X; 
    Y = L.Y;

    % Allow matdat.mat to override maturities (for consistency across scripts)
    if isfield(L,'maturities'), maturities = L.maturities; J = numel(maturities); end
end

% --------------------------- Sample Selection ----------------------------
% Early sample window used only for:
% - Benchmark CS regressions (not for network training below).
t = 1:400;                         % Training window indices for CS OLS
T = numel(t);                      % Number of observations in CS window

% ------------------------ Campbell–Shiller OLS ---------------------------
% Baseline linear model for each maturity:
%   y_i(t) = alpha_i + beta_i * slope_i(t) + eps_i(t),
% where slope_i(t) corresponds to X(:,i) here (first J columns of X).
CS = zeros(J,2,'single');          % [alpha_i, beta_i] for i = 1,...,J
O  = ones(T,1,'single');           % Column of ones for intercept
for i = 1:J
    % Solve (O, slope_i) * [alpha_i; beta_i] = y_i via OLS on the CS window
    CS(i,:) = ([O, single(X(t,i))] \ single(Y(t,i))).';
end

% ----------------------- Network Hyperparameters -------------------------
% hyper struct is passed into b2bLayer, which constructs:
% - Input layer with "inputSize" features (columns of X).
% - Shared hidden layers with "numHiddenLayers" and "numNeurons".
% - Output heads for each maturity that start from CS prediction and learn
%   residual structure on top (design handled inside b2bLayer).
hyper.inputSize       = size(X,2);   % Number of predictors D
hyper.outputSize      = J;           % Number of outputs J (maturities)
hyper.numNeurons      = 24;          % Width of each hidden layer
hyper.numHiddenLayers = 2;           % Number of fully connected hidden layers

% ------------------------- Build Network Object --------------------------
rng(RNG_SEED,'twister');           % Fix all RNG usage prior to network build
dlnet = b2bLayer(maturities, hyper, CS, VERBOSE);
% dlnet:
% - Implements forward(dlnet, dlX, 'Outputs', outNames{k}) for each maturity.
% - Contains learnable parameters for shared layers + residual heads.

% ------------------------- CPU/GPU Selection -----------------------------
% Select computation device based on "useGPUMode" flag and GPU availability.
switch lower(useGPUMode)
    case 'gpu'
        useGPU = true;
    case 'cpu'
        useGPU = false;
    otherwise
        % 'auto' mode: use GPU only if canUseGPU is present and returns true
        useGPU = (exist('canUseGPU','file') && canUseGPU);
end
if useGPU
    % Move all learnables inside dlnet to GPU
    dlnet = dlupdate(@gpuArray, dlnet);
end

% ----------------------------- Accelerate --------------------------------
% dlaccelerate builds an optimized graph for the given function handle,
% which can substantially speed up repeated dlfeval calls during training.
if exist('dlaccelerate','file')
    modelGradients_acc = dlaccelerate(@modelGradients);   % For training loop
    jointMSE_acc       = dlaccelerate(@jointMSE);         % For validation
else
    modelGradients_acc = @modelGradients;
    jointMSE_acc       = @jointMSE;
end

% -------------------------- Output Head Names ----------------------------
% Each output head is named "sumXX", where XX is maturity in years.
% These names must match the internal layer naming convention in b2bLayer.
outNames = arrayfun(@(m) sprintf('sum%02d', m), maturities, 'UniformOutput', false);

% ----------------------- Train / Validation Split ------------------------
% Simple chronological split:
% - First (1 - valFrac) of sample for training (Xtr, Ytr)
% - Remaining tail for validation (Xva, Yva)
valFrac = 0.2;
Tall    = size(X,1);               % Total time-series length
Ttr     = floor((1 - valFrac) * Tall);
Xtr     = single(X(1:Ttr,:));      % [Ttr x D] training predictors
Ytr     = single(Y(1:Ttr,1:J));    % [Ttr x J] training targets
Xva     = single(X(Ttr+1:end,:));  % [Tva x D] validation predictors
Yva     = single(Y(Ttr+1:end,1:J));% [Tva x J] validation targets

% --------------------------- Optimizer (Adam) ----------------------------
% Standard Adam hyperparameters for stochastic optimization.
numEpochs = 1000;                  % Maximum number of passes over training set
miniBatch = 64;                    % Mini-batch size in time dimension
learnRate = 1e-3;                  % Base learning rate
beta1     = 0.9;                   % Exponential decay for first moment
beta2     = 0.999;                 % Exponential decay for second moment

% Adam state (initialized empty; updated inside training loop)
trailingAvg   = [];
trailingAvgSq = [];
iter          = 0;                 % Global iteration counter (across epochs)

% Batched data layout conversions:
% - Input: X in [T x D] -> dlarray [D x B] with format 'CB'
% - Target: Y in [T x J] -> dlarray [J x B] with format 'CB'
toDLX = @(A) dlarray(A.', 'CB');   % [T x D] -> [D x B]
toDLY = @(A) dlarray(A.', 'CB');   % [T x J] -> [J x B]

% ----------------------- Monitoring: Loss Curves -------------------------
% Pre-allocate arrays for epoch-level train/validation MSE
epochTrain = nan(numEpochs,1,'single');
epochVal   = nan(numEpochs,1,'single');

% Simple live plot of training/validation loss over epochs
f  = figure('Name','Training Monitor','NumberTitle','off');
ax = axes(f); grid(ax,'on'); hold(ax,'on');
xlabel(ax,'Epoch'); ylabel(ax,'MSE');
hTrain = plot(ax, nan, nan, '-o', 'DisplayName','Train');
hVal   = plot(ax, nan, nan, '-s', 'DisplayName','Validation');
legend(ax, 'show', 'Location','northeast');

% Text overlay for wall-clock time
t0 = tic;  % elapsed time start
hElapsed = text(ax, 0.02, 0.95, 'Elapsed: 0.0 s', ...
    'Units','normalized','HorizontalAlignment','left','VerticalAlignment','top');

% ------------------------------ Training ---------------------------------
% Reset RNG again before shuffling:
% - Ensures that mini-batch composition is reproducible across runs.
rng(RNG_SEED,'twister');           
bestVal = inf;                     % Best (lowest) validation loss so far
bestNet = dlnet;                   % Snapshot of best-performing network

for epoch = 1:numEpochs
    % Random permutation of training indices for this epoch (stochasticity)
    idx = randperm(Ttr);
    batchLossSum = 0; 
    batchCnt     = 0;

    % Iterate over mini-batches in time index space
    for s = 1:miniBatch:Ttr
        iter = iter + 1;
        b = idx(s:min(s+miniBatch-1, Ttr));  % Current batch indices

        % Convert current batch to dlarray with [D x B] and [J x B] layout
        dlX = toDLX(Xtr(b,:));
        dlY = toDLY(Ytr(b,:));
        if useGPU
            dlX = gpuArray(dlX); 
            dlY = gpuArray(dlY);
        end

        % Compute joint MSE loss and gradients w.r.t. learnables
        [L, G] = dlfeval(modelGradients_acc, dlnet, dlX, dlY, outNames);

        % Single Adam update step on all learnable parameters in dlnet
        [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, G, ...
            trailingAvg, trailingAvgSq, iter, learnRate, beta1, beta2);

        % Accumulate batch loss for epoch-level average
        batchLossSum = batchLossSum + gather(extractdata(L));
        batchCnt     = batchCnt + 1;
    end

    % Record training loss for this epoch
    epochTrain(epoch) = batchLossSum / batchCnt;

    % Evaluate validation loss (no gradient) on full validation sample
    dlXv = toDLX(Xva);
    dlYv = toDLY(Yva);
    if useGPU
        dlXv = gpuArray(dlXv); 
        dlYv = gpuArray(dlYv);
    end
    valLoss = dlfeval(jointMSE_acc, dlnet, dlXv, dlYv, outNames);
    epochVal(epoch) = gather(extractdata(valLoss));

    % Track best model according to validation MSE
    if valLoss < bestVal
        bestVal = valLoss;
        bestNet = dlnet;
    end

    % Update training monitor plot
    set(hTrain, 'XData', 1:epoch, 'YData', epochTrain(1:epoch));
    set(hVal,   'XData', 1:epoch, 'YData', epochVal(1:epoch));
    title(ax, sprintf('Train MSE: %.4g | Validation MSE: %.4g', epochTrain(epoch), epochVal(epoch)));
    set(hElapsed, 'String', sprintf('Elapsed: %.1f s', toc(t0)));
    drawnow;
end

% ------------------------------ Save Model -------------------------------
% Store the best validation model in CPU memory for portability.
bestNetCPU = dlupdate(@gather, bestNet);
save(fullfile(dataDir,'best_dlnet.mat'), 'bestNetCPU','bestVal','hyper','maturities');

% ============================= Functions =================================
function [L, G] = modelGradients(dlnet, dlX, dlY, outNames)
% modelGradients:
% - Inputs:
%   dlnet    : dlnetwork with shared hidden layers and multiple heads.
%   dlX      : dlarray [D x B] of predictors (batch).
%   dlY      : dlarray [J x B] of targets (batch).
%   outNames : cell array of head names, length J.
% - Output:
%   L        : scalar dlarray, joint MSE across maturities and batch.
%   G        : structure of gradients w.r.t. dlnet.Learnables.
%
% Loss definition:
%   For each maturity k:
%       yk_hat = forward(dlnet, dlX, 'Outputs', outNames{k})
%       MSE_k  = mean( (yk_hat - yk_true).^2 )
%   Joint loss:
%       L = (1/J) * sum_k MSE_k
%
    Jloc = size(dlY,1);            % Number of outputs (maturities) in batch
    acc  = dlarray(0, 'CB');       % Accumulator for sum of MSEs across heads

    for k = 1:Jloc
        % Forward pass for maturity k; output shape [1 x B]
        yk = forward(dlnet, dlX, 'Outputs', outNames{k});  % [1 x B]

        % Residual for maturity k
        d  = yk - dlY(k,:);                              % [1 x B]

        % Accumulate mean squared error over batch
        acc = acc + mean(d.^2, 'all');
    end

    % Average across maturities
    L = acc / Jloc;

    % Backpropagate through dlnet.Learnables
    G = dlgradient(L, dlnet.Learnables);
end

function L = jointMSE(dlnet, dlX, dlY, outNames)
% jointMSE:
% - Same loss as in modelGradients, but without computing gradients.
% - Used for validation: reports joint MSE across maturities and batch.
%
% Inputs:
%   dlnet    : trained (or current) dlnetwork object.
%   dlX      : dlarray [D x B], predictors.
%   dlY      : dlarray [J x B], targets.
%   outNames : cell array of head names, length J.
%
% Output:
%   L        : scalar dlarray, joint MSE across maturities and batch.
%
    Jloc = size(dlY,1);
    acc  = dlarray(0, 'CB');

    for k = 1:Jloc
        % Forward pass for head k
        yk = forward(dlnet, dlX, 'Outputs', outNames{k});
        d  = yk - dlY(k,:);
        acc = acc + mean(d.^2, 'all');
    end

    % Average over heads
    L = acc / Jloc;
end
