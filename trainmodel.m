function [SVMModel1,SVMModel2,SVMModel3,SVMModel4] = trainmodel(p,r)
% Input: 
%       r: a vector of threshold, with dimension m x n_model
%       p: normalized innovation sequence with dimension Ntrain x m

if(nargin<2)
    r = [1.5; 2.5; 5]; % I AM HERE!
end

Ntrain = size(p,2); % number of training samples

pbar = vecnorm(p,1); % Compute L-1 norm of normalized innovation sequence

r1 = r(1); r2 = r(2); r3 = r(3); 
p1 = 1-nnz(pbar<=r1)/Ntrain;
p2 = 1-nnz(pbar<=r2)/Ntrain;
p3 = 1-nnz(pbar<=r3)/Ntrain;
p4 = nnz(pbar>r3)/Ntrain;
SVMModel1 = trainOCSVM(p,p1);
SVMModel2 = trainOCSVM(p,p2);
SVMModel3 = trainOCSVM(p,p3);
SVMModel4 = trainOCSVM(p,p4);
