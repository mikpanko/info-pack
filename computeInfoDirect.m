function [I, Ipt] = computeInfoDirect(X, Y, rangeX, rangeY)

% Compute mutual information (MI) using the direct method alone and with
% Panzeri-Treves bias correction
% Algorithm following Magri et al. (2009) "A toolbox for the fast information
% analysis of multiple-site LFP, EEG and spike train recordings"
% ----------------------------------------
% X - discretized signal / neural data
% Y - discretized response / intended BMI targets
% rangeX - vector of numbers of different possible values of signal X along each dimension/channel
% rangeY - number of different possible values of response Y

% set parameters
nTr = length(Y);
nCh = size(X, 2);
if ~exist('rangeY', 'var')
    rangeY = max(Y);
end
if ~exist('rangeX', 'var')
    rangeX = nan(1, nCh);
    for c = 1:nCh
        rangeX(c) = max(X(:, c));
    end
end
trgs = [1:rangeY];
nTrg = rangeY;

% pre-compute x*log(x) values
xlogx = zeros(nTr+1, 1);
for i = 1:nTr
    xlogx(i+1) = i*log2(i);
end

% compute entropies H(R) and H(R|S)
HR = 0;
HRS = zeros(1, nTrg);
CR = zeros([rangeX, 1]);
CRS = zeros([rangeX, nTrg]);
if nCh>1
    xy = cell(1, nCh+1);
    for t = 1:nTr
        for i = 1:nCh
            xy{i} = X(t, i);
        end
        xy{nCh+1} = Y(t);
        Cr = sum(CRS(xy{1:nCh}, :));
        HR = HR + xlogx(Cr+2) - xlogx(Cr+1);
        HRS(Y(t)) = HRS(Y(t)) + xlogx(CRS(xy{:})+2) - xlogx(CRS(xy{:})+1);
        CRS(xy{:}) = CRS(xy{:}) + 1;
    end
else
    for t = 1:nTr
        x = X(t);
        y = Y(t);
        HR = HR + xlogx(CR(x)+2) - xlogx(CR(x)+1);
        CR(x) = CR(x) + 1;
        HRS(y) = HRS(y) + xlogx(CRS(x,y)+2) - xlogx(CRS(x,y)+1);
        CRS(x, y) = CRS(x,y) + 1;
    end
end

% normalize entropies H(R) and H(R|S)
HR = log2(nTr)-1/nTr*HR;
nTrS = hist(Y, trgs);
pTrg = nTrS/nTr;
HRS = pTrg.*(log2(nTrS)-1./nTrS.*HRS);
HRS = sum(HRS(~isnan(HRS)));

% compute mutual information
I = HR - HRS;

% perform simplified Panzeri-Treves bias correction
CR = sum(CRS, nCh+1);
Ipt = I - (nnz(CRS)-nTrg-nnz(CR)+1)/(2*nTr*log(2));
