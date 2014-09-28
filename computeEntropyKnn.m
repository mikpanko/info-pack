function H = computeEntropyKNN(X, k)

% Compute K-nn entropy estimate 
% ----------------------------------------
% X - signal / neural data
% k - number of neighbors to use

% set parameters
[N, D] = size(X);

% compute distance between samples: dist(i,j) - distance between samples X(i,:) and X(j,:)
tmp = X*X';
squares = diag(tmp);
dist = sqrt( repmat(squares,1,N) + repmat(squares',N,1) - 2*tmp);

% sort distances
sortedDist = sort(dist, 'ascend');
minK = sum(sortedDist==0);
k = max(k, max(minK));

% compute entropy
H = nan(size(k));
for i = 1:length(k)
    H(i) = D/N*sum(log2(sortedDist(k(i)+1,:))) + log2(pi^(D/2)/gamma(D/2+1)) - psi(k(i))/log(2) + log2(N);
end
