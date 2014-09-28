function decoderResults = decodeData(X, Y, params)

% Decode data and perform cross-validation
% Based on Scott Brincat's code for BMI neural signal decoding
% ----------------------------------------
% X - signal / neural data
% Y - response / intended BMI targets
% params - parameters structure

% Fields of the structure "params":
% 'crossValType' - type of cross-validation: 'none', 'leave-one-out' or 'N-fold' (where N is a positive integer)
% 'decType' - type of decoder: 'lda', 'knn', 'mlr' (multinomial logistic regression) or 'map' (maximum a-posteriori Baysian inference)
% 'LDAdiscrimType' - (for 'lda' decoder only) discriminat type (see MATLAB help for "ClassificationDiscriminant" class)
% 'numNeighbors' - (for 'knn' decoder only) number of neighbors to use
% 'distance' - (for 'knn' decoder only) distance metric to use (see MATLAB help for "ClassificationKNN" class)
% 'spkPreds' - (for 'map' decoder only) logical vector flagging which predictors are based on spike count vs something else (ie, LFP power)

% % load features
% Ftr = load('C:\!analysis\CS20120505\CS20120505-features-lfp-[80-500Hz]-[0-750ms]-SEF.mat');
% features = Ftr.features(:, find(Ftr.params.chs==49));
% 
% % load events
% loadParams.session = Ftr.params.session;
% loadParams.dataType = 'evt';
% Evt = loadData(loadParams);
% 
% % select conditions
% conds = selectConditions(Ftr.params.session, Evt, 'target-direction');
% conds = conds(Ftr.params.trials);

% set parameters
if ~exist('params', 'var')
    params = struct;
end
if ~isfield(params, 'crossValType')
    params.crossValType = '10-fold';
end
if ~isfield(params, 'decType')
    params.decType = 'lda';
end
if strcmp(params.decType, 'lda') && ~isfield(params, 'LDAdiscrimType')
    params.LDAdiscrimType = 'pseudolinear';
end
if strcmp(params.decType, 'knn') && ~isfield(params, 'numNeighbors')
    params.numNeighbors = 100;
end
if strcmp(params.decType, 'knn') && ~isfield(params, 'distance')
    params.distance = 'cityblock';
end
if strcmp(params.decType, 'map') && ~isfield(params, 'spkPreds')
    params.spkPreds = 0;
end
decoderResults.params = params;

% decode data without cross-validation
nTr = size(X, 1);
switch params.decType
    case 'lda'
        decoderResults.prediction = LDAdecoder(X, Y, X, params.LDAdiscrimType);
    case 'knn'
        decoderResults.prediction = KNNdecoder(X, Y, X, params.numNeighbors, params.distance);
    case 'mlr'
        decoderResults.prediction = MLRdecoder(X, Y, X);
    case 'map'
        decoderResults.prediction = MAPdecoder(X, Y, X, params.spkPreds);
end
testY = Y;

% perform cross-validation if requested
if ~strcmp(params.crossValType, 'none')
    decoderResults.trainAccuracy = 100*sum(decoderResults.prediction==testY)/nTr;
        
    % set the size of testing sets
    if strcmp(params.crossValType, 'leave-one-out')
        nTestSets = nTr;
        testSetSize = 1;
    elseif strcmp(params.crossValType(end-4:end), '-fold')
        nTestSets = str2double(params.crossValType(1:end-5));
        testSetSize = ceil(nTr/nTestSets);
    else
        error('CANNOT RECOGNIZE CROSS-VALIDATION TYPE!');
    end
    
    % shuffle data
    rng('shuffle');
    shuffledTr = randperm(nTr);
    
    % decode data
    for n = 1:nTestSets
        testSet = (testSetSize*(n-1)+1) : min(testSetSize*n, nTr);
        trainX = X;
        trainX(shuffledTr(testSet), :) = [];
        testX = X(shuffledTr(testSet), :);
        trainY = Y;
        trainY(shuffledTr(testSet)) = [];
        switch params.decType
            case 'lda'
                decoderResults.prediction(testSet, 1) = LDAdecoder(trainX, trainY, testX, params.LDAdiscrimType);
            case 'knn'
                decoderResults.prediction(testSet, 1) = KNNdecoder(trainX, trainY, testX, params.numNeighbors, params.distance);
            case 'mlr'
                decoderResults.prediction(testSet, 1) = MLRdecoder(trainX, trainY, testX);
            case 'map'
                decoderResults.prediction(testSet, 1) = MAPdecoder(trainX, trainY, testX, params.spkPreds);
        end
    end
    testY = Y(shuffledTr);
    
end

% compute overall cross-validation decoder accuracy (percent correct choices over all test trials)
decoderResults.accuracy = 100*sum(decoderResults.prediction==testY)/nTr;

% calculate confusion matrix(actual,predicted) showing number of trials where decoder predicts given response target for each actual target
decoderResults.confusionMtx = confusionmat(testY, decoderResults.prediction);

% calculate cross-validated decoder accuracy (pct correct) for each actual response target
decoderResults.tgtAccuracy = 100*diag(decoderResults.confusionMtx)./sum(decoderResults.confusionMtx,2);
