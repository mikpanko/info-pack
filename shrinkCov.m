function covMtx = shrinkCov(X, method)

% Estimate covariance matrix using one of shrinkage methods based on formulas
% from: Chen et al (2010) "Shrinkage Algorithms for MMSE Covariance Estimation"
% ----------------------------------------
% X - signal / neural data
% method - shrinkage method to estimate covariance

% Possible values of "method":
% 'lw' - Ledoit-Wolf estimator
% 'rblw' - Rao-Blackwell Ledoit-Wolf estimator
% 'oas' - oracle approximating shrinkage estimator
% '' - (empty) regular covariance matrix without shrinkage

% set parameters
n = size(X, 1);
p = size(X, 2);
if ~exist('method', 'var')
    method = 'oas';
end

% compute supplementary matrices and traces
S = cov(X, 1);
trS = trace(S);
trS2 = trace(S'*S);
F = trS/p*eye(p);

% compute shrinkage coefficient
switch method
    
    case 'lw'
        rho = 0;
        for i = 1:n
            rho = rho + (norm(X(i,:)'*X(i,:)-S, 'fro'))^2;
        end
        rho = rho/(n^2*(trS2-trS^2/p));
        
    case 'rblw'
        rho = ((n-2)/n*trS2+trS^2)/(n+2)/(trS2-trS^2/p);
        
    case 'oas'
        rho = ((1-2)/p*trS2+trS^2)/((n+1-2)/p)/(trS2-trS^2/p);
        
    case ''
        rho = 0;
        
end
rho = min(rho, 1);

% compute shrinkage covariance
covMtx = (1-rho)*S + rho*F;
