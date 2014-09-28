function H = computeEntropyGaussianMixtureMonteCarlo(means, covs, weights, nSamples)

% Estimate entropy of a Gaussian mixture using Monte Carlo random sampling
% ----------------------------------------
% means - array of means of Gaussians in the mixture
% covs - array of covariance matrices of Gaussians in the mixture
% weights - vector of weights of Gaussians in the mixture
% nSamples - number of Monte Carlo samples

% reset random number generator
rng('shuffle')

% convert data if it is in matrix not cell array form
if ~iscell(means)
    means = num2cell(means, 2);
    covs = num2cell(covs, [1,2]);
    weights = num2cell(weights);
end

% set parameters
L = length(means); % number of Gaussians in the mixture
D = length(means{1}); % number of dimensions
if ~exist('nSamples', 'var')
    nSamples = 5000; % number of Monte Carlo samples
end

% find sqrtm(inv(covs)) and normal coefficients for each Gaussian
normCoefs = nan(L, 1);
for i = 1:L
    sqrtInvCovs{i} = sqrtm(inv(covs{i}));
    normCoefs(i) = ((2*pi)^D*det(covs{i}))^(-0.5);
end

H = 0;
for i = 1:L
    
    smpls = mvnrnd(means{i}, covs{i}, nSamples);
    
    gaussMixture = 0;
    for j = 1:L
        gaussMixture = gaussMixture + weights{j}*normCoefs(j)*exp(-0.5*sum(((smpls-ones(nSamples,1)*means{j})*sqrtInvCovs{j}).^2,2));
    end
    
    H = H - weights{i}*mean(log2(gaussMixture));

end
