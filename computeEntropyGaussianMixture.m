function H = computeEntropyGaussianMixture(means, covs, weights, method, X, Y)

% Estimate entropy of a Gaussian mixture using formulas from Huber et al.
% (2008) "On Entropy Approximation for Gaussian Mixture Random Vectors"
% ----------------------------------------
% means - array of means of Gaussians in the mixture
% covs - array of covariance matrices of Gaussians in the mixture
% weights - vector of weights of Gaussians in the mixture
% method - integer corresponding to formula to use to compute entropy
% X - (optional) signal / neural data
% Y - (optional) response / intended BMI targets

% Possible values of "method":
% 0 - lower bound
% 1 - upper bound
% 2 - 2nd order Taylor expansion

% convert data if it is in matrix not cell array form
if ~iscell(means)
    means = num2cell(means, 2);
    covs = num2cell(covs, [1,2]);
    weights = num2cell(weights);
end

% set parameters
L = length(means); % number of Gaussians in the mixture
D = length(means{1}); % number of dimensions
nlog2pie = D*log2(2*pi*exp(1));

switch method
    
    % lower bound
    case 0
        H = 0;
        for i = 1:L
            tmp = 0;
            for j = 1:L
                tmp = tmp + weights{j}*mvnpdf(means{i}, means{j}, covs{i}+covs{j});
            end
            H = H - weights{i}*log2(tmp);
        end
        
    % upper bound
    case 1
        
        % initial entropy upper bound on all Gaussian components
        H = 0;
        for i = 1:L
            H = H - weights{i}*(log2(weights{i})-0.5*(nlog2pie+log2(det(covs{i}))));
        end
        
        % generate large amount of Gaussian data if it is not provided
        if ~exist('X', 'var')
            n = 100000;
            X = nan(n*L, D);
            Y = nan(n*L, 1);
            for i = 1:L
                X(((i-1)*n+1):(i*n), :) = mvnrnd(means{i}, covs{i}, n);
                Y(((i-1)*n+1):(i*n)) = i;
            end
        end

        Hall = nan(L, 1);
        Hall(L) = H;
        compSet = num2cell(1:L);
        compSetWeights = weights;
        for i = L:-1:2
            
            % compute Runnall's distance between Gaussian components
            B = nan(i, i);
            for j = 1:i
                ij = ismember(Y, compSet{j});
                for k = (j+1):i
                    ik = ismember(Y, compSet{k});
                    B(j, k) = 0.5 * ( (compSetWeights{j}+compSetWeights{k})*log2(det(cov(X(ij|ik,:)))) - ...
                       compSetWeights{j}*log2(det(cov(X(ij,:)))) - compSetWeights{k}*log2(det(cov(X(ik,:)))) );
                end
            end
            
            % merge 2 Gaussian components with the smallest Runnall's distance
            Bmin = min(B(:));
            [j, k] = find(B==Bmin, 1);
            compSet{j} = [compSet{j}, compSet{k}];
            compSet(k) = [];
            compSetWeights{j} = compSetWeights{j} + compSetWeights{k};
            compSetWeights(k) = [];
            
            % compute entropy upper bound on new components
            Hall(i-1) = 0;
            for j = 1:(i-1)
                ij = ismember(Y, compSet{j});
                Hall(i-1) = Hall(i-1) - compSetWeights{j}*(log2(compSetWeights{j})-0.5*(nlog2pie+log2(det(cov(X(ij,:))))));
            end
            
        end
        
        % find minimum entropy upper bound
        H = min(Hall);
        
    % 2nd order Taylor expansion
    case 2
               
        % compute inv(covs)
        covsInv = {};
        for i = 1:L
            covsInv{i} = inv(covs{i});
        end
        
        H = 0;
        for i = 1:L
            
            % compute pdf(means_i) and grad(pdf(means_i))
            wpgauss = nan(L, 1);
            pmeansi = 0;
            gradpmeansi = zeros(D, 1);
            for j = 1:L
                wpgauss(j) = weights{j}*mvnpdf(means{i}, means{j}, covs{j});
                pmeansi = pmeansi + wpgauss(j);
                gradpmeansi = gradpmeansi + covsInv{j}*(means{i}-means{j})'*wpgauss(j);
            end
            
            % 0th order term
            H = H - weights{i}*log2(pmeansi);
            
            % compute F(means_i)
            F = zeros(D, D);
            for j = 1:L
                dif = (means{i} - means{j})';
                F = F + covsInv{j}*(-1/pmeansi*dif*gradpmeansi'+dif*(covsInv{j}*dif)'-eye(D))*wpgauss(j);
            end
            F = F/pmeansi;
            
            % 2nd order term
            H = H - weights{i}/2/log(2)*sum(sum(F.*covs{i}));

        end
        
end
