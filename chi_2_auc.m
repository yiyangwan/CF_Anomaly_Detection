function [auc_prc, auc_roc] = chi_2_auc(p, s_test, s_f_test, config, idm_para, s_f, anomaly_idx, varargin)
%      - 'numThresh'    : Specify the (maximum) number of score intervals.
%                         Generally, splits are made such that each
%                         interval contains about the same number of sample
%                         lines.
num_thresh = 100;

optargin = size(varargin, 2);
stdargin = nargin - optargin;

i = 1;
while (i <= optargin)
    if (strcmp(varargin{i}, 'numThresh'))
        if (i >= optargin)
            error('argument required for %s', varargin{i});
        else
            num_thresh = varargin{i+1};
            i = i + 2;
        end
    end
end

sen = zeros(num_thresh, 1);
ppv = zeros(num_thresh, 1);
fpRate = zeros(num_thresh, 1);

qvals = (1:(num_thresh-1))/num_thresh;
rr = [0 quantile(p.chi,qvals)];
for i = 1:length(rr)
%     disp(i)
    config.r = rr(i);
    [~,err_temp,~]    = CfFilter(s_test, s_f_test, config, idm_para, s_f, 'printStatus',false);
    err_temp             = logical(err_temp');
    TP     = nnz(err_temp(anomaly_idx==1));  % true positive
    FP     = nnz(err_temp(anomaly_idx==0));  % false positive
    TN     = nnz(~err_temp(anomaly_idx==0)); % true negative
    FN     = nnz(~err_temp(anomaly_idx==1)); % false negative

    sen(i)  = TP / (TP + FN);
    ppv(i)  = TP/(TP+FP);
    spec = TN/(TN+FP);
    fpRate(i) = 1-spec;
end
auc_roc = trapz([0;flip(fpRate);1], [0;flip(sen);1]);
auc_prc = trapz([0;flip(sen);1], [1; flip(ppv); 0]);
end
