function newSig = transformSignal(X, transformType)

% Transform signal
% ----------------------------------------
% X - signal / neural data
% transformType - type of signal transformation

epsilon = 1e-20;

switch transformType
    
    case 'sqrt'
        newSig = sqrt(X);
        
    case 'log'
        rng('shuffle');
        idx = (X==0);
        newSig = log(X);
        newSig(idx) = log(epsilon) + randn(sum(idx(:)), 1);
        
    case 'cubic-root'
        newSig = X.^(1/3);
        
    case 'none'
        newSig = X;
        
end
