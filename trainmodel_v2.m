function [SVMModel1,SVMModel2,SVMModel3,SVMModel4] = trainmodel_v2(p,r)
% Input: 
%       r: a vector of threshold, with dimension m x n_model
%       p: normalized innovation sequence with dimension Ntrain x m

SVMModel1 = trainOCSVM(p,r(1));
SVMModel2 = trainOCSVM(p,r(2));
SVMModel3 = trainOCSVM(p,r(3));
SVMModel4 = trainOCSVM(p,r(4));
