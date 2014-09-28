function [info, infoBtsp] = computeInfoIBTB(X, Y, params)

% Compute information measures on multi-dimensional responses using
% Information Breakdown Toolbox (IBTB)
% The toolbox is introduced in Magri et al. (2009) "A toolbox for the fast
% information analysis of multiple-site LFP, EEG and spike train recordings"
% ----------------------------------------
% X - signal / neural data
% Y - response / intended BMI targets
% params - parameters structure

% Fields of the structure "params":
% "metric" - information metric to compute (see "information.m" in IBTB)
% "method" - method of computing the metric (see "information.m" in IBTB)
% "bias" - bias correction method to use (see "information.m" in IBTB)
% "dscBinNum" - number of bins to use for data discretization
% "btspNum" - number of bootstrap samples (see "information.m" in IBTB)
% verbose" - boolean on printing details of computation (see "information.m" in IBTB)

% set parameters
if ~exist('params', 'var')
    params = struct;
end
if ~isfield(params, 'metric')
    params.metric = 'Ish';
end
if ~isfield(params, 'method')
    params.method = 'dr';
end
if ~isfield(params, 'bias')
    params.bias = 'pt';
end
if ~isfield(params, 'dscBinNum')
    if any(rem(X(:), 1))
        params.dscBinNum = length(unique(Y))*10;
    else
        params.dscBinNum = 0;
    end
end
if ~isfield(params, 'btspNum')
    params.btspNum = 0;
end
if ~isfield(params, 'verbose')
    params.verbose = false;
end

% discretize data
if params.dscBinNum>0
    X = discretizeData(X, params.dscBinNum, 'eq-space');
end

% compute response matrix
[R, nt] = buildr2(Y', X');

% calculate information measures
opts.nt = nt;
opts.method = params.method;
opts.bias = params.bias;
opts.btsp = params.btspNum;
opts.verbose = params.verbose;
I = information(R, opts, params.metric);
info = I(1);
if params.btspNum>0
    infoBtsp = I(2:end);
else
    infoBtsp = [];
end
