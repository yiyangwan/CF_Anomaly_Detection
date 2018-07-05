clear

load rawdata.mat % info of the leading vehicle = s
load following_state.mat % info of the following vehicle = s_f
raw_data   = s;
% Config data structure====================================================
config.OCSVM = false; % if true, then use OCSVM instead of Chi-square detector
config.OCSVM_threshold = [3; 4; 5]; % OCSVM model threshold for training
config.R = diag([1,0.5]); % observation noise covariance
config.Q = diag([0.8,0.05]); % process noise covariance
config.H = eye(2); % observation matrix
config.N = 15; % Time window length for AdEKF
config.r = 5; % Chi-square detector parameter
config.delta_t = 0.1; % sensor sampling time interval in seconds
config.tau = 0.7; % time delay
config.N_ocsvm = 15; % Time window length for OCSVM

% IDM CF model parameter===================================================
idm_para.a = 0.73; % maximum acceleration
idm_para.b = 1.67; % comfortable deceleration
idm_para.sigma = 4; % acceleration exponent 
idm_para.s0 = 2; % minimum distance (m)
idm_para.T = 1.5; % safe time headway (s)
idm_para.v0 = 24; % desired velocity (m/s)
idm_para.a_max = 0.1; % max acceleration of random term 
idm_para.a_min = -0.1; % max deceleration of random term
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
x_f = s_f(:,1);
v_f = s_f(:,2);
save('following_state.mat','s_f')

%% Run experiments
s= s(5:end,:)';
s_f = s_f(5:2000,:)';

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

AnomalyConfig.percent = 0.02;
AnomalyConfig.anomaly_type = {'Noise','Bias','Drift'};
AnomalyConfig.dur_length = 50;
AnomalyConfig.NoiseVar = diag(sqrt([50,10]));
AnomalyConfig.BiasVar = diag(sqrt([50,10]));
AnomalyConfig.DriftMax = [50;10];

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
    s = s_test';
    s_f = s_f_test';
    
%     [x1,y1,~,auc1] = perfcurve(AnomalyConfig.index,p.score,1);


else
    [shat,err,p] = CfFilter(s_test, s_f_test, config, idm_para, s_f);
    err = logical(err');
end

%% Generate summary

anomaly_idx = AnomalyConfig.index(1,:) | AnomalyConfig.index(2,:);
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

Summary1.configuration = config;
Summary1.car_following_para = idm_para;
Summary1.anomaly = AnomalyConfig;
Summary1.results = table(metric_name,metric_values);
summary(Summary1.results)

filename = 'Summary1.mat';
save(filename,'Summary1')


%% Plotting
close all
[m,n] = size(s_fa);

for i = 1:n
    s_fa(:,i) = s_fa(:,i) + config.R*randn(m,1);
end

xx = 1:length(err);

x_l = s_la(1,:);
v_l = s_la(2,:);

x_f1 = s_fa(1,:);
v_f1 = s_fa(2,:);
subplot(411)
plot(v_l);hold on; plot(v_f1);legend('leading-raw','following-raw'); ylim([0,40]);
subplot(412)
plot(x_l);hold on; plot(x_f1);legend('leading-raw','following-raw');

x_f2 = shat(1,:);
v_f2 = shat(2,:);

subplot(413)
plot(v_l);hold on; plot(v_f2);
plot(xx(err),v_f2(err),'b*');
plot(xx(AnomalyConfig.index(2,:)),v_f2(AnomalyConfig.index(2,:)),'d')
legend('leading-raw','following-filtered'); ylim([0,40]);

subplot(414)
plot(x_l);hold on; plot(x_f2);legend('leading-raw','following-filtered');
plot(xx(AnomalyConfig.index(1,:)),x_f2(AnomalyConfig.index(1,:)),'d');
figure
h_x = histogram(p0.innov(1,:));
h_x.BinWidth = 0.2;
hold on
mean_location = mean(p0.innov(1,:))
h_v = histogram(p0.innov(2,:)); legend('vehicle location','vehicle speed')
hold off
h_v.BinWidth = 0.2;
xlim([-10,10]);
mean_speed = mean(p0.innov(2,:))

% figure
% plot(x1,y1), legend('ROC curve of OCSVM');
% auc1