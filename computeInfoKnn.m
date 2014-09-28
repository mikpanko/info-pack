function I = computeInfoKNN(X, Y, k)

% Compute mutual information using K-nn method
% ----------------------------------------
% X - signal / neural data
% Y - response / intended BMI targets
% k - number of neighbors to use

% set parameters
nTr = length(Y);
trgs = unique(Y);
nTrg = length(trgs);
nTrS = hist(Y, trgs);
pTrg = nTrS/nTr;

% estimate entropies H(R|S)
HRS = zeros(nTrg, length(k));
for trg = 1:nTrg
    HRS(trg, :) = computeEntropyKNN(X(Y==trgs(trg),:), k);
end
HRS = repmat(pTrg', [1,length(k)]).*HRS;
HRS = sum(HRS);

% estimate entropy H(R)
HR = computeEntropyKNN(X, k);

% compute mutual information
I = HR - HRS;
