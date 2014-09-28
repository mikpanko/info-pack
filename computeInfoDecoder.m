function I = computeInfoDecoder(confusionMtx, infoType)

% Compute mutual information (MI) extracted from neural signals by decoder
% ----------------------------------------
% confusionMtx - decoder confusion matrix with dimensions [# of actual targets, # of decoded targets]
% infoType - type of decoder results to use for MI calculation: "confusionMtx" or "accuracy"

nTrg = size(confusionMtx, 1);
nTr = sum(sum(confusionMtx));

% MI based on decoder confusion matrix
% Formula from Quian Quiroga & Panzeri (2009) "Extracting information from
% neuronal populations: information theory and decoding approaches"
if strcmp(infoType, 'confusionMtx')
    Pssp = confusionMtx / nTr;
    Ps = sum(Pssp, 2);
    Psp = sum(Pssp, 1);
    
    Ps = repmat(Ps, [1, nTrg]);
    Psp = repmat(Psp, [nTrg, 1]);
    I = Pssp.*log2(Pssp./Ps./Psp);
    I = sum(sum(I(~isnan(I))));
end

% MI based on decoder accuracy
% Formula from Waldert et al. (2009) "A review on directional information
% in neural signals for brain-machine interfaces"
if strcmp(infoType, 'accuracy')
    ca = 100/nTrg; % chance accuracy
    da = 100*sum(diag(confusionMtx))/nTr; % decoder accuracy
    I = da/100*log2(da/ca)+(100-da)/100*log2((100-da)/(100-ca));
end
