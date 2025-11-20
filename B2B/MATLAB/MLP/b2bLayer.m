function dlnet = b2bLayer(mats, hyper, CS_coefs, VERBOSE)
% b2bLayer: Constructs the CS + Residual MLP Graph
% - mats: Array of maturities
% - hyper: Hyperparameters struct
% - CS_coefs: [J x 2] matrix of Frozen CS coefficients [alpha, beta]

    inputSize = hyper.inputSize;
    numNeurons = hyper.numNeurons;
    numHiddenLayers = hyper.numHiddenLayers;
    outputSize = numel(mats);
    
    BiasInit    = 'zeros';
    WeightsInit = 'glorot';
    
    % Naming
    slpnames = arrayfun(@(x) sprintf('slp%02d', x), mats, 'UniformOutput', false);
    CSnames  = arrayfun(@(x) sprintf('CS%02d', x), mats, 'UniformOutput', false);
    fcnames  = arrayfun(@(x) sprintf('fc%d', x), 1:numHiddenLayers, 'UniformOutput', false);
    actnames = arrayfun(@(x) sprintf('elu%d', x), 1:numHiddenLayers, 'UniformOutput', false);
    resnames = arrayfun(@(x) sprintf('res%02d', x), mats, 'UniformOutput', false);
    
    % Input Layer
    input = featureInputLayer(inputSize, 'Name', 'input');
    lgraph = layerGraph(input); 
    
    % Picker & CS Frozen Paths
    % Assumption: X = [S(1..J), OtherVars...]
    % The picker creates a one-hot vector to select the j-th Slope
    picker = [eye(outputSize), zeros(outputSize, inputSize - outputSize)];
    
    for j = 1:outputSize
        % 1. Slope Picker (Identity weight, frozen)
        slpWeightInit = @(sz) single(picker(j,:)); 
        slp = fullyConnectedLayer(1, ...
            'Name', slpnames{j}, ...
            'BiasLearnRateFactor', 0, 'WeightLearnRateFactor', 0, ...
            'BiasInitializer', 'zeros', 'WeightsInitializer', slpWeightInit);
    
        % 2. CS Affine Transform (Alpha + Beta*S, frozen)
        alpha_val = CS_coefs(j,1);
        beta_val  = CS_coefs(j,2);
        
        csBiasInit   = @(sz) single(alpha_val) * ones(sz, 'single');
        csWeightInit = @(sz) single(beta_val) * ones(sz, 'single');
    
        cs = fullyConnectedLayer(1, ...
            'Name', CSnames{j}, ...
            'BiasLearnRateFactor', 0, 'WeightLearnRateFactor', 0, ...
            'BiasInitializer', csBiasInit, 'WeightsInitializer', csWeightInit);
    
        lgraph = addLayers(lgraph, slp);
        lgraph = addLayers(lgraph, cs);
        lgraph = connectLayers(lgraph, 'input', [slpnames{j} '/in']);
        lgraph = connectLayers(lgraph, slpnames{j}, [CSnames{j}  '/in']);
    end
    
    % Residual MLP Trunk (Learnable)
    sharedTrunk = [];
    for k = 1:numHiddenLayers
        sharedTrunk = [sharedTrunk; ...
            fullyConnectedLayer(numNeurons, 'Name', fcnames{k}, ...
            'BiasInitializer', BiasInit, 'WeightsInitializer', WeightsInit); ...
            eluLayer('Name', actnames{k})];
    end
    lgraph = addLayers(lgraph, sharedTrunk);
    lgraph = connectLayers(lgraph, 'input', [fcnames{1} '/in']);
    
    % Residual Heads & Summation
    for j = 1:outputSize
        % Head
        resHead = fullyConnectedLayer(1, 'Name', resnames{j}, ...
            'BiasInitializer', BiasInit, 'WeightsInitializer', WeightsInit);
        lgraph = addLayers(lgraph, resHead);
        lgraph = connectLayers(lgraph, actnames{end}, [resnames{j} '/in']);
        
        % Sum (CS + Residual)
        sumname = sprintf('sum%02d', mats(j));
        addL = additionLayer(2, 'Name', sumname);
        lgraph = addLayers(lgraph, addL);
        
        lgraph = connectLayers(lgraph, CSnames{j}, [sumname '/in1']);
        lgraph = connectLayers(lgraph, resnames{j}, [sumname '/in2']);
    end
    
    dlnet = dlnetwork(lgraph);
    
    if VERBOSE
        plot(lgraph); title('CS + Residual MLP Architecture');
    end
end