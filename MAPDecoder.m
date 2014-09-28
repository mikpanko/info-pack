function predictedTgts = MAPdecoder(trainingX, trainingY, testX, spkPreds)

% by Scott Brincat

% function [predictedTgts, popVectors] = bayesMLDecoder(trainingX, trainingY, testX, tgtDirections, decoder, spkPreds, probTransform)
% 
% Bayesian inference or MAP-inference (Maximum A Posteriori)/Maximum likelihood based decoders
%  Both are very similar, so use this same function and branch out below.  Both calculate the conditional probability
%  of the neural response observed in each test trial, given each potential stimulus (target direction), under the 
%  response statistics observed across training trials (currently assuming Poisson spike counts, Gaussian LFP powers). 
%  MAP/ML then finds the stimulus (target direction) that makes the observed test response most likely (by Bayes thm.,
%  equivalent to finding the most likely stimulus, given the response, if targets are equally probable).  
%  Bayesian inference performs a weighted average of the conditional probabilities of response given the stimulus to
%  generate an estimated saccade direction (which is then discretized to the nearest target).
%  Note: continuous estimate is part of Bayesian inference calculation, but kludged for MAP/ML by interpolating posterior prob's
% Based on Dayan & Abbott _Theoretical Neuroscience_ ch3.3 (p.101-7)
% 
% INPUTS:
% decoder         String indicating which type of decoder to use here (both seem to have similar performance):
%                 'map'/'ml': Maximum A Posteriori-inference / Maximum Likelihood decoder (equivalent for equiprobable tgts) or
%                 'bayes'   : Bayesian-inference decoder
% spkPreds        Logical vector (1,nPredictors) flagging which predictors are based on spike count vs something else (ie, LFP power)
%                 Calculations for spike count predictors assume Poisson probability dist'n, while all others assume Gaussian dist'n
% probTransform (optional) Function handle used to transform (eg, log) decoder-predicted probabilities before popVector computation

[nTestTrls,nP]  = size(testX);  
nTgts           = length(unique(trainingY));
tgtIndexes      = trainingY - min(trainingY) + 1;     % Convert trial tgt numbers to tgt indices (1:nTgts) if not already

% Calculate across-training-trial response mean and std dev for each target, for each predictor
tgtMeans  = nan([1 nP nTgts]);
tgtStds   = nan([1 nP nTgts]);
for iTgt = 1:nTgts
  tgtMeans(1,:,iTgt)  = mean(trainingX(tgtIndexes == iTgt,:), 1);
  tgtStds(1,:,iTgt)   = std(trainingX(tgtIndexes == iTgt,:), 0, 1);    
end

% Calculate P(r|s) -- conditional probability of observed test responses given each possible stimulus, for each test trial
Prs = nan([nTestTrls nP nTgts]);
% Assume Poisson probability dist'n for spike counts 
% todo: make assumed dist'ns flexible? tho, normal clearly doesn't work as well for spike counts...
if any(spkPreds) 
  ndx   = spkPreds;                                         % Indices of all spike-count predictors
  lamda = repmat(tgtMeans(1,ndx,:), [nTestTrls 1 1]);       % Poisson rate parameter (mean spike count for each tgt; rep'd to nTrials)
  k     = repmat(testX(:,ndx), [1 1 nTgts]);                % Observed spike counts for each trial, units (rep'd to nTgts)
  Prs(:,ndx,:) = (lamda.^k) .* exp(-lamda) ./ factorial(k); % Poisson probability of observed count for each tgt, 
end                                                         %  given mean training-data rate for each tgt
% Assume Gaussian probability dist'n for LFP power (should really log-transform power to normalize)
if any(~spkPreds)
  ndx   = ~spkPreds;                                        % Indices of all non-spike predictors
  mu    = repmat(tgtMeans(1,ndx,:), [nTestTrls 1 1]);       % Gaussian mean parameter (mean response for each tgt; rep'd to nTrials)
  sig   = repmat(tgtStds(1,ndx,:), [nTestTrls 1 1]);        % Gaussian stddev parameter (stddev of response for each tgt; rep'd to nTrials)
  x     = repmat(testX(:,ndx), [1 1 nTgts]);                % Observed responses for each trial and predictor (rep'd to nTgts)
  z     = (x - mu) ./ sig;
  Prs(:,ndx,:) = exp(-0.5*z.^2) .* (1./(sqrt(2*pi).*sig));  % Gaussian probability of observed response for each tgt, given mean training-data response for each tgt
  % Kludge: set Prs=1 for any entries w/ sig==0 so they don't affect final product of prob's, since prob is kinda undefined here                  
  Prs(sig==0) = 1;
end

% Probability of observed population response, given each possible stimulus, on each test trial = 
%  product of probabilities of all individual predictors (assumes independence of predictors to 
%  greatly simplify computation, though it's clearly not strictly true)
PRs = reshape( prod(Prs, 2), [nTestTrls nTgts] );           % Note: PRs array size(nTestTrls,nTgts)

% MAP (Maximum a Posteriori) inference (Dayan & Abbott eqn's 3.30) -- select stimulus with maximum posterior probability 
% Equivalent to Maximum Likelihood inference if stimuli (saccade directions) are equally probable.
% Note: assumes all tgts have equal probability, also ignores p(R) term, which is independent of stimulus
[~,predictedTgts] = max(PRs, [], 2);      % Select most probable stimulus

end
