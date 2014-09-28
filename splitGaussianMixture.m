function [newMeans, newCovs, newWeights] = splitGaussianMixture(means, covs, weights, maxRep, maxEig)

% Split Gaussians in Gaussian mixture into more Gaussians to reduce
% variance in each using formula from Huber et al. (2008) "On Entropy
% Approximation for Gaussian Mixture Random Vectors"
% ----------------------------------------
% means - array of means of Gaussians in the mixture
% covs - array of covariance matrices of Gaussians in the mixture
% weights - vector of weights of Gaussians in the mixture
% maxRep - maximum number of splits
% maxEig - maximum tolerated variance of individual Gaussians

if ~exist('maxRep', 'var')
    maxRep = 20;
end

if ~exist('maxEig', 'var')
    maxEig = 0;
end

if ~iscell(means)
    means = num2cell(means, 2);
    covs = num2cell(covs, [1,2]);
    weights = num2cell(weights);
end

% set splitting parameters
spM = [-1.4131205233, -0.44973059608, 0.44973059608, 1.4131205233];
spC = [0.51751260421, 0.51751260421, 0.51751260421, 0.51751260421].^2;
spW = [0.12738084098, 0.37261915901, 0.37261915901, 0.12738084098];
spN = length(spM);

% set dimension parameters
nGauss = length(covs);
nDim = size(covs{1}, 1);

% prepare eigenvectors and values of covariance matrices
V = nan(nDim, nDim, nGauss);
D = nan(nDim, nDim, nGauss);
for i = 1:nGauss
    [V(:,:,i), D(:,:,i)] = eig(covs{i});
end
[nEig, iEig] = max(D(:));
disp(['max var = ', num2str(nEig)]);
iRep = 0;

% repeat splitting until reaching maximum tolerated variance or number of repetitions 
while (iRep<maxRep) && (nEig>maxEig)
    
    % select component to split
    [iD, iD, iG] = ind2sub(size(D), iEig);
    
    % split the component
    m = zeros(spN, nDim);
    m(:, iD) = sqrt(nEig)*spM';
    newM = repmat(means{iG}, [spN,1]) + m;
    newD = repmat(D(:,:,iG), [1,1,spN]);
    newD(iD, iD, :) = shiftdim(spC*nEig, 1);
    newC = nan(nDim, nDim, spN);
    for i = 1:spN
        newC(:, :, i) = V(:,:,iG)*newD(:,:,i)*V(:,:,iG)';
    end
    newW = weights{iG}*spW;
    newV = repmat(V(:,:,iG), [1,1,spN]);
    
    % replace with new components
    means(iG) = [];
    covs(iG) = [];
    weights(iG) = [];
    V(:, :, iG) = [];
    D(:, :, iG) = [];
    newIdx = nGauss:(nGauss+spN-1);
    for i = 1:length(newIdx)
        means{newIdx(i)} = newM(i, :);
        covs{newIdx(i)} = newC(:, :, i);
        weights{newIdx(i)} = newW(i);
    end
    V(:, :, newIdx) = newV;
    D(:, :, newIdx) = newD;
    nGauss = nGauss + spN - 1;
    
    % update repetitions and largest variance component
    iRep = iRep + 1;
    [nEig, iEig] = max(D(:));
    
end

% output results
newMeans = means;
newCovs = covs;
newWeights = weights;
