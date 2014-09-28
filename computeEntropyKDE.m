function H = computeEntropyKDE(X)

% Estimate entropy sample using a kernel density estimator with Silverman's
% rule of thumb for the bandwidth parameter
% ----------------------------------------
% X - signal / neural data

% set parameters
N = size(X, 1); % number of samples
D = size(X, 2); % number of dimensions
sigma = (det(shrinkCov(X,'oas'))^(1/D))/(N^(1/5));

% estimate entropy
H = 0;
for i = 1:N

    dist = X - repmat(X(i,:), [N,1]);
    dist = sqrt(sum(dist.^2, 2));
    dist(i) = [];
    Htmp = normpdf(dist, 0, sigma);
    Htmp = log2(mean(Htmp));
    if ~isinf(Htmp)
        H = H - Htmp;
    end
    
end
H = H/N;
