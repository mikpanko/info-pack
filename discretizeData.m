function discreteData = discretizeData(X, nBins, method)

% Discretize data into equal sized and linearly spaced bins
% ----------------------------------------
% X - signal / neural data
% nBins - number of discrete bins
% method - discretization method

if ~exist('method', 'var')
    method = 'eq-space';
end

switch method
    
    % equally spaced discretization
    case 'eq-space'
        discreteData = nan(size(X));
        for c = 1:size(X,2)
            step = range(X(:,c))/nBins;
            offset = min(X(:,c));
            discreteData(:,c) = floor((X(:,c)-offset)/step)+1;
            discreteData(discreteData(:,c)==(nBins+1),c) = nBins;
        end
        
    % equally populated discretization
    case 'eq-popul'
        [nTr, nCh] = size(X);
        discreteData = nan(size(X));
        for c = 1:nCh
            edges = quantile(X(:,c), [0:1/nBins:1]);
            [~, discreteData(:,c)] = histc(X(:,c), edges);
            discreteData(discreteData(:,c)==(nBins+1),c) = nBins;
        end
       
end
