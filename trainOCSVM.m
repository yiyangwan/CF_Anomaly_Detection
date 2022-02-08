function SVMModel = trainOCSVM(chi,p)
% Input:
%       chi: training samples from one distribution with dimension m x n_sample
%       p: percentage of outliers used for training OCSVM at scale from 0
%          to 1
% Output:
%       SVMModel = OCSVM model

train   = chi';
[n,~]   = size(train); % n: # of training samples; m: # of dimension
label   = ones(n,1);

%%------------------------ Train One-class SVM ----------------------------

SVMModel = fitcsvm(train,label,'KernelScale','auto',...
    'Standardize',true,'OutlierFraction',p);
% SVMModel = fitSVMPosterior(SVMModel);
