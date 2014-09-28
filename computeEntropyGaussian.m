function [H, HconfInt] = computeEntropyGaussian(X, errorFormula, alpha)

% Compute mutual information of a Gaussian with confidence intervals based
% on formulas from Djauhari (2009) "Asymptotic Distributions of Sample
% Covariance Determinant"
% ----------------------------------------
% X - signal / neural data
% errorFormula - integer corresponding to the confidence interval formula to use
% alpha - alpha value to use with the confidence interval formula

% Possible values of "errorFormula":
% 0 - no confidence intervals
% 1 - formula from Corollary to Theorem 1 in the article
% 2 - formula from Theorem 2 in the article

if nargin<3
    alpha = 0.05;
end
if nargin<2
    errorFormula = 0;
end

% set parameters
nTr = size(X, 1);
nCh = size(X, 2);

% compute det(cov(R))
detX = det(cov(X));

% compute entropy H(R)
nlog2pie = nCh*log2(2*pi*exp(1));
H = 0.5*(nlog2pie+log2(detX));

% compute confidence intervals
switch errorFormula
    
    % no confidence intervals
    case 0
        HconfInt = [nan, nan];
    
    % formula from Corollary to Theorem 1 (simpler)
    case 1
        detMin = norminv(alpha/2, detX, sqrt(2*nCh/(nTr-1)*detX^2));
        detMax = norminv(1-alpha/2, detX, sqrt(2*nCh/(nTr-1)*detX^2));
        Hmin = 0.5*(nlog2pie+log2(detMin));
        Hmax = 0.5*(nlog2pie+log2(detMax));
        HconfInt = [Hmin, Hmax];
        
    % formula from Theorem 2 (more precise)
    case 2
        if (errorFormula==2)
            b1 = 1;
            b2 = 1;
            for i = 1:nCh
                b1 = b1*(nTr-i)/(nTr-1);
                b2 = b2*(nTr-i+2)/(nTr-1);
            end
            b2 = b1*(b2-b1);
        end
        detMin = norminv(alpha/2, detX/b1, sqrt(detX^2*b2/b1^2));
        detMax = norminv(1-alpha/2, detX/b1, sqrt(detX^2*b2/b1^2));
        Hmin = 0.5*(nlog2pie+log2(detMin));
        Hmax = 0.5*(nlog2pie+log2(detMax));
        HconfInt = [Hmin, Hmax];
        
end
