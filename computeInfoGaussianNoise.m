function I = computeInfoGaussianNoise(X, Y, method, shrinkMethod)

% Estimate mutual information using Gaussian model with bias reduction using
% incremental noisy dimensions
% ----------------------------------------
% X - signal / neural data
% Y - response / intended BMI targets
% methods - a cell array of Gaussian methods to compute MI
% shrinkMethod - method to estimate Gaussian covariances

% Possible values of "method":
% "int" - numerical integration
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

% set parameters
[nSmpl, nDim] = size(X);

% estimate MI by adding dimensions one-by-one
I = 0;
for n = 1:nDim
    X0 = X(:, 1:n);
    I1 = computeInfoGaussian2(X0, Y, method, shrinkMethod);
    X0(:, end) = randn(nSmpl, 1);
    I2 = computeInfoGaussian2(X0, Y, method, shrinkMethod);
    I = I + I1 - I2;
end
