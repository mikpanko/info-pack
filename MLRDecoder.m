function [predictedTgts, B] = MLRdecoder(trainingX, trainingY, testX)
 
% Multinomial logistic regression-based decoder -- Regress x and y using
% generalized linear model with logit link function and model allowing for
% multiple categories
% by Scott Brincat

B           = mnrfit(trainingX, trainingY, 'interactions','on');  % Fit MNR model to training data (returns model coeff's)
P           = mnrval(B, testX);                   % Used fitted coeffs to calc prob's for each tgt on each trial
[~,predictedTgts] = max(P,[],2);                  % Method 1: Predicted tgt = tgt w/ max probability on each trial

end
