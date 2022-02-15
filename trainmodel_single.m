function [SVMModel1] = trainmodel_single(p,r)
% Input: 
%       r: a vector of threshold, with dimension m x n_model
%       p: normalized innovation sequence with dimension Ntrain x m

if(nargin<2)
    r = [1.5; 2.5; 5]; % I AM HERE!
end

Ntrain = size(p,2); % number of training samples

pbar = vecnorm(p,1); % Compute L-1 norm of normalized innovation sequence

r1 = r(1);
p1 = 1-nnz(pbar<=r1)/Ntrain;
SVMModel1 = trainOCSVM(p,p1);
