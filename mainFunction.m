function [Summary] = mainFunction(config,idm_para,AnomalyConfig,raw_data)

rng(20); % control the random generator so that it generates predictable random number

%% Generate baseline data
[x_l,v_l] = data_process(raw_data); % get leading vehicle location x_l, speed v_l and acceleration a_l

% Generate following vehicle location x_f, speed v_f and acceleration a_l based on a
% car-following model 

x0 = 10; % initial location of following vehicle
v0 = 1; % initial speed of following vehicle

tau = config.tau; % human/sensor reaction time delay with unit "s"
delta_t = config.delta_t; % sampling time interval with unit "s"

t  = ceil(tau/delta_t); % time delay in discrete state-transition model

s_f = cf_model(x_l,v_l,x0,v0,delta_t,t,tau,idm_para);

%% Run experiments
s= raw_data(7:end,:)';
s_f = s_f(7:2000,:)';

%==========================================================================
%   AnomalyConfig: 
%       .index: index of anomaly
%       .percent: percent of anomaly in scale [0,1]
%       .anomaly_type: list of anomaly types, should be a list in choice of
%       'Instant','Bias','Drift'
%       .dur_length: the max duration of anomaly
%       .NoiseVar: Noise type anomaly standard covariance matrix with dimension m x m
%       .BiasVar: Bias type anomaly covariance matrix with dimension m x m
%       .DriftVar: Drift type anomaly max value


% Generate anomalous data
[s_la, s_fa, AnomalyConfig] = generateAnomaly(s, s_f, AnomalyConfig); 
AnomalyIdx = AnomalyConfig.index; % ground truth

s_test = s_la; s_f_test = s_fa; % test dataset

%% Run Models
% Generate statistics for baseline data
if(config.OCSVM)
    config.OCSVM = false;
    [~,~,p0] = CfFilter(s, s_f, config, idm_para, s_f); 
    config.OCSVM = true;
else
    [~,~,p0] = CfFilter(s, s_f, config, idm_para, s_f); 
end
    
% Train several OCSVM models with different sensitivity levels
if(config.OCSVM)
    [SVMModel1,SVMModel2,SVMModel3,SVMModel4] = trainmodel(p0.innov,config.OCSVM_threshold);
    
    config.SVMModel1 = SVMModel1;
    config.SVMModel2 = SVMModel2;
    config.SVMModel3 = SVMModel3;
    config.SVMModel4 = SVMModel4;
    
 % Test OCSVM
    
    [shat,err,p] = CfFilter(s_test, s_f_test, config, idm_para, s_f);    
    err = logical(err');
%     s = s_test';
%     s_f = s_f_test';
    
%     [x1,y1,~,auc1] = perfcurve(AnomalyConfig.index,p.score,1);


else
    [shat,err,p] = CfFilter(s_test, s_f_test, config, idm_para, s_f);
    err = logical(err');
end

%% Generate summary

anomaly_idx = AnomalyIdx(1,:) | AnomalyIdx(2,:);
TP     = nnz(err(anomaly_idx==1)); % true positive
FP     = nnz(err(anomaly_idx==0)); % false positive
TN     = nnz(~err(anomaly_idx==0)); % true negative
FN     = nnz(~err(anomaly_idx==1)); % false negative

f1   = 2*TP/(2*TP+FP+FN);
acc  = (TP+TN)/(TN+FN+FP+TP);
spec = TN/(TN+FP);
sen  = TP / (TP + FN);
ppv  = TP/(TP+FP);

metric_name = {'TP';'FP';'TN';'FN';'F1';'Accuracy';'Specificity';'Sensitivity'...
    ;'Precision'};
metric_values = [TP;FP;TN;FN;f1;acc;spec;sen;ppv];

Summary.configuration = config;
Summary.car_following_para = idm_para;
Summary.anomaly = AnomalyConfig;
Summary.results = table(metric_name,metric_values);

end