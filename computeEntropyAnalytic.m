function H = computeEntropyAnalytic(type, params)

% Compute analytical entropy of various multivariate distributions
% ----------------------------------------
% type - type of analytical distribution
% params - parameters of the distribution

switch type
    
    case 'gauss'
        detCov = det(params.cov);
        prodSigmas = prod(diag(params.cov));
        p = size(params.cov, 1);
        H = 0.5*(p*log2(2*pi*exp(1))+log2(detCov/prodSigmas));
        
    case 't'
        detCov = det(params.cov);
        prodSigmas = prod(diag(params.cov));
        p = size(params.cov, 1);
        n = params.df;
        H = 0.5*log2(detCov/prodSigmas)+p/2*log2(n*pi)+log2(beta(p/2,n/2)/gamma(p/2))+(n+p)/2/log(2)*(psi((n+p)/2)-psi(n/2));
        
    case 'parabolic'
        nCh = size(params.sigma, 1);
        detHalfSigma = det(params.sigma^(1/2));
        
        % compute parameter of the multivariate parabolic distribution
        B = ( gamma(nCh/2+2) / pi^(nCh/2) / detHalfSigma )^(2/(nCh+2));
                
        % generate a large sample of precursor parabolic data
        nTr = 10000000;
        ftrs = rand(nTr, nCh);
        a = 3/4/B^(3/2);
        b = a*B;
        delta0 = a*b;
        u = (-1-1i*sqrt(3))/2;
        delta1 = 3*a^2*(ftrs-1/2);
        C = ((delta1+sqrt(delta1.^2-4*delta0^3))/2).^(1/3);
        ftrs = real(-1/a*(u*C+delta0/u./C));
        
        % compute PDFs of generated data
        ftrsSq = ftrs.^2;
        P1 = (B - sum(ftrsSq, 2))*detHalfSigma;
        P0 = prod(B-ftrsSq, 2);
        P = P1./P0;
        
        % mold generated data to parabolic distribution
        Prnd = max(P)*rand(length(P), 1);
        P = P(P>=Prnd);
       
        % use Monte Carlo method to estimate true entropy
        H = -mean(log2(P));
        
end
