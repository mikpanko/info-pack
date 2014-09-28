function [Y, X, means, covs] = generateData(nTr, distrType, paramMethod, params)

% Generate multi-variate Gaussian data
% ----------------------------------------
% nTr - number of trials/samples to generate
% distrType - analytic distribution to sample data from
% paramMethod - method to generate data
% params - parameters structure based on the 'paramMethod'

% Possible values of "distrType":
% 'gauss' - Gaussian distribution
% 't' - t distribution
% 'parabolic' - parabolic distribution

% Possible values of "paramMethod":
% 'specify' - specify 'means' and 'covs' in directly (as fields in 'params')
% 'load' - load 'means' and 'covs' variables from 'filename' (field in 'params')
% 'test' - provide 'nTrg', 'nCh' and 'var' (as fields in 'params') and generate test means and covs 

% reset random number generator
rng('shuffle');

% get distribution parameters: means, covs, nTrg and nCh
switch paramMethod
    
    % get from input parameters
    case 'specify'
        means = params.means;
        covs = params.covs;
        [nTrg, nCh] = size(means);
        
    % load from file
    case 'load'
        load(params.filename);
        [nTrg, nCh] = size(means);
        
    % pick by hand
    case 'test'
        nTrg = params.nTrg;
        nCh = params.nCh;
        means = zeros(nTrg, nCh);
        for i = 1:nTrg
            means(i, 1) = cos(2*pi/nTrg*i);
            means(i, 2) = sin(2*pi/nTrg*i);
        end
        covs = repmat(diag(params.var*ones(1,nCh)), [1,1,nTrg]);
        
end

% generate targets
Y = repmat([1:nTrg]', [ceil(nTr/nTrg),1]);
Y = Y(1:nTr);

% generate random correlated variables
switch distrType
    
    % Gaussian distribution
    case 'gauss'
        X = mvnrnd(means(Y,:), covs(:,:,Y), nTr);
        
    % Student's t distribution (heavy tails)
    case 't'
        X = nan(nTr, nCh);
        for i = 1:nTrg
            idx = i:nTrg:nTr;
            X(idx, :) = mvtrnd(covs(:,:,i), params.df, length(idx)).*repmat(sqrt(diag(covs(:,:,i))'), [length(idx),1]) + repmat(means(i,:), [length(idx),1]);
        end
        
    % parabolic distribution (short tails)
    case 'parabolic'
        X = nan(nTr, nCh);
        for i = 1:nTrg
            halfCov = squeeze(covs(:,:,i))^(1/2);
            detHalfCov = det(halfCov);
            idx = i:nTrg:nTr;
            nTr0 = 1e6;
            idxIdx = 0;
            
            % compute parameter of the multivariate parabolic distribution
            B = ( gamma(nCh/2+2) / pi^(nCh/2) / detHalfCov )^(2/(nCh+2));
            
            while idxIdx<length(idx)
                
                % generate a large sample of precursor parabolic data
                ftrs0 = rand(nTr0, nCh);
                a = 3/4/B^(3/2);
                b = a*B;
                delta0 = a*b;
                u = (-1-1i*sqrt(3))/2;
                delta1 = 3*a^2*(ftrs0-1/2);
                C = ((delta1+sqrt(delta1.^2-4*delta0^3))/2).^(1/3);
                ftrs0 = real(-1/a*(u*C+delta0/u./C));
                
                % compute PDFs of generated data
                ftrs0sq = ftrs0.^2;
                P1 = (B - sum(ftrs0sq, 2))*detHalfCov;
                P0 = prod(B-ftrs0sq, 2);
                P = P1./P0;
                
                % mold generated data to desired parabolic distribution
                Prnd = max(P)*rand(length(P), 1);
                idx1 = (P>=Prnd);
                idx2 = 1:nTr0;
                idx2 = idx2(idx1);
                nTrTmp = min(length(idx)-idxIdx, length(idx2));
                idx2 = idx2(1:nTrTmp);
                ftrs0 = ftrs0(idx2, :);
                X(idx([1:nTrTmp]+idxIdx), :) = ftrs0*halfCov' + repmat(means(i,:), [nTrTmp,1]);
                idxIdx = idxIdx + nTrTmp;
                
            end
            
        end
        
end
