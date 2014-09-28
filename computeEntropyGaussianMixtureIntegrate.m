function H = computeEntropyGaussianMixtureIntegrate(means, covs, weights, N)

% Compute entropy of a 2D Gaussian mixture by numerical integration
% ----------------------------------------
% means - array of means of Gaussians in the mixture
% covs - array of covariance matrices of Gaussians in the mixture
% weights - vector of weights of Gaussians in the mixture
% N - (optional) number of st.dev. to include in the integration region (large values can make integration unstable)

% convert data if it is in matrix not cell array form
if ~iscell(means)
    means = num2cell(means, 2);
    covs = num2cell(covs, [1,2]);
    weights = num2cell(weights);
end

% set parameters
L = length(means); % number of Gaussians in the mixture
D = 2; % number of dimensions
if ~exist('N', 'var')
    N = 8; % number of st.dev. to include in the integration region
end

% set integration boundaries
maxEig = nan(L, 1);
for i = 1:L
    maxEig(i) = eigs(covs{i}, 1);
end
means0 = cell2mat(means);
minX = min(means0-N*repmat(maxEig, [1,D]), [], 1);
maxX = max(means0+N*repmat(maxEig, [1,D]), [], 1);

% compute entropy
entropyFunc2 = @(x,y)entropyFunc(x,y,means,covs,weights);
H = -integral2(entropyFunc2, minX(1), maxX(1), minX(2), maxX(2));

end

function z = entropyFunc(x, y, means, covs, weights)

sizex = size(x);
p = zeros(numel(x), 1);
for i = 1:length(means)
    p = p + weights{i}*mvnpdf([x(:),y(:)], means{i}, covs{i});
end
z = p.*log2(p);
z = reshape(z, sizex);

end
