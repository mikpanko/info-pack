function [covMtx, estEigVl] = mestreCov(X)

% Estimate covariance matrix using Mestre's method optimizing eigenvalue
% estimation accuracy
% Based on formula from: Mestre (2008) "Improved estimation of eigenvalues
% and eigenvectors of covariance matrices using their sample estimates"
% UNTESTED!
% ----------------------------------------
% X - signal / neural data

% set parameters
N = size(X, 1); % number of samples
M = size(X, 2); % number of dimensions

% compute sample covariance, eigenvalues, and eigenvectors
smplCov = cov(X);
[smplEigVc, smplEigVl] = eig(smplCov);
smplEigVl = diag(smplEigVl);

% solve for mu(m)
x = sym('var');
eq = smplEigVl(1)/(smplEigVl(1)-x);
for m = 2:M
    eq = eq + smplEigVl(m)/(smplEigVl(m)-x);
end
eq = eq - N;
mu = double(solve(eq));
mu = real(mu);
mu = sort(mu);

% compute robust eigenvalue estimates
estEigVl = N*(smplEigVl - mu);

% compute theta(m,k)
tmp = zeros(M, M);
for m = 1:M
    for k = 1:M
        if (k~=m)
            tmp(m, k) = smplEigVl(m)/(smplEigVl(k)-smplEigVl(m)) - mu(m)/(smplEigVl(k)-mu(m));
        end
    end
end
theta = -tmp;
for m = 1:M
    theta(m, m) = 1 + sum(tmp(:,m));
end

% compute robust eigenvector projection matrix P(m)
P = zeros(M, M, M);
for m = 1:M
    for k = 1:M
        P(:, :, m) = P(:,:,m) + theta(m,k)*smplEigVc(:,k)*smplEigVc(:,k)';
    end
end

% compute Mestre's robust covariance estimate
covMtx = zeros(M, M);
for m = 1:M
    covMtx = covMtx + estEigVl(m)*squeeze(P(:,:,m));
end
