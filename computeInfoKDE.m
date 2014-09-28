function I = computeInfoKDE(X, Y)

% Compute mutual information using kernel density estimator
% ----------------------------------------
% X - signal / neural data
% Y - response / intended BMI targets

% set parameters
nTr = length(Y);
trgs = unique(Y);
nTrg = length(trgs);
nTrS = hist(Y, trgs);
pTrg = nTrS/nTr;
    
% compute entropies H(R|S)
HRS = zeros(1, nTrg);
for t = 1:nTrg
    HRS(t) = computeEntropyKDE(X(Y==trgs(t), :));
end
HRS = pTrg.*HRS;
HRS = sum(HRS(~isnan(HRS)));

% compute entropy H(R)
HR = computeEntropyKDE(X);

% compute mutual information
I = HR - HRS;
