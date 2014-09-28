function I = computeInfoGaussian(X, Y, methods, shrinkMethod)

% Compute mutual information (MI) of a Gaussian mixture
% ----------------------------------------
% X - signal / neural data
% Y - response / intended BMI targets
% methods - a cell array of Gaussian methods to compute MI
% shrinkMethod - method to estimate Gaussian covariances

% Possible members of the cell array "methods" to compute H(R):
% "int" - numerical integration (works only for 2D Gaussians)
% "gauss" - single Gaussian approximation to Gaussian mixture
% "lower" - lower bound on Gaussian mixture
% "upper" - upper bound on Gaussian mixture
% "taylor" - Taylor extension of Gaussian mixture
% "taylorSplit" - Taylor extension of Gaussian mixture with Gaussian splitting
% "monteCarlo" - Monte Carlo Gaussian mixture method

% Possible values of "shrinkMethod":
% 'lw' - Ledoit-Wolf estimator
% 'rblw' - Rao-Blackwell Ledoit-Wolf estimator
% 'oas' - oracle approximating shrinkage estimator
% '' - (empty) regular covariance matrix without shrinkage
% 'mestre' - Mestre method

if nargin<4
    shrinkMethod = 'rblw';
end
if nargin<3
    methods = 'monteCarlo';
end
if ischar(methods)
    methods = {methods};
end

if isvector(Y)
    % trial data
    
    % set parameters
    nTr = length(Y);
    nCh = size(X, 2);
    trgs = unique(Y);
    nTrg = length(trgs);
    nTrS = hist(Y, trgs);
    pTrg = nTrS/nTr;
    
    % compute mean(R|S), cov(R|S), det(cov(R|S)), and det(cov(R))
    means = nan(nTrg, nCh);
    covs = nan(nCh, nCh, nTrg);
    detRS = nan(nTrg, 1);
    for trg = 1:nTrg
        data = X(Y==trg, :);
        means(trg, :) = mean(data);
        if strcmp(shrinkMethod, 'mestre')
            covs(:, :, trg) = mestreCov(data);
        else
            covs(:, :, trg) = shrinkCov(data, shrinkMethod);
        end
        detRS(trg) = det(covs(:,:,trg));
    end
    if ismember('gauss', methods)
        if strcmp(shrinkMethod, 'mestre')
            detR = det(mestreCov(X));
        else
            detR = det(shrinkCov(X, shrinkMethod));
        end
    end
    
else
    % mean and covs data

    % set parameters
    nCh = size(X, 2);
    nTrg = size(X, 1);
    pTrg = 1/nTrg*ones(1, nTrg);
    
    % compute det(cov(R|S)) and det(cov(R))
    means = X;
    covs = Y;
    detRS = nan(nTrg, 1);
    for trg = 1:nTrg
        detRS(trg) = det(covs(:,:,trg));
    end
    if ismember('gauss', methods)
        n = 100000;
        X = nan(n*nTrg, nCh);
        Y = nan(n*nTrg, 1);
        for trg = 1:nTrg
            X(((trg-1)*n+1):(trg*n), :) = mvnrnd(means(trg,:), covs(:,:,trg), n);
            Y(((trg-1)*n+1):(trg*n)) = trg;
        end
        if strcmp(shrinkMethod, 'mestre')
            detR = det(mestreCov(X));
        else
            detR = det(shrinkCov(X, shrinkMethod));
        end
    end

end

% compute entropies H(R|S) and H(R)
nlog2pie = nCh*log2(2*pi*exp(1));
HRS = zeros(1, nTrg);
for trg = 1:nTrg
    HRS(trg) = 0.5*(nlog2pie+log2(detRS(trg)));
end
HRS = pTrg.*HRS;
HRS = sum(HRS(~isnan(HRS)));

% compute entropy H(R)
weights = pTrg;
if (ismember('int', methods) && (nCh==2))
    HR.int = computeEntropyGaussianMixtureIntegrate(means, covs, weights);
else
    HR.int = nan;
end
if ismember('gauss', methods)
    HR.gauss = 0.5*(nlog2pie+log2(detR));
else
    HR.gauss = nan;
end
if ismember('lower', methods)
    HR.lower = computeEntropyGaussianMixture(means, covs, weights, 0);
else
    HR.lower = nan;
end
if ismember('upper', methods)
    HR.upper = computeEntropyGaussianMixture(means, covs, weights, 1, X, Y);
else
    HR.upper = nan;
end
if ismember('taylor', methods)
    HR.taylor = computeEntropyGaussianMixture(means, covs, weights, 2);
else
    HR.taylor = nan;
end
if ismember('taylorSplit', methods)
    [meansSpl, covsSpl, weightsSpl] = splitGaussianMixture(means, covs, weights, 20, 0);
    HR.taylorSplit = computeEntropyGaussianMixture(meansSpl, covsSpl, weightsSpl, 2);
else
    HR.taylorSplit = nan;
end
if ismember('monteCarlo', methods)
    HR.monteCarlo = computeEntropyGaussianMixtureMonteCarlo(means, covs, weights);
else
    HR.monteCarlo = nan;
end

% compute mutual information
I.int = HR.int - HRS;
I.gauss = HR.gauss - HRS;
I.lower = HR.lower - HRS;
I.upper = HR.upper - HRS;
I.taylor = HR.taylor - HRS;
I.taylorSplit = HR.taylorSplit - HRS;
I.monteCarlo = HR.monteCarlo - HRS;
if (length(methods)==1)
    I = I.(methods{1});
end
