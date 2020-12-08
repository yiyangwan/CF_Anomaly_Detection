
clear

filePath = 'dataset\'; % dataset location
load(strcat(filePath,'testdata.mat')) % info of the leading vehicle = s for testing n_sample * m
load(strcat(filePath,'rawdata.mat')) % info of the leading vechicle = s_train for training n_sample * m
% load(strcat(filePath,'following_state.mat')) % info of the following vehicle = s_f
raw_data = s_train;
% s = s_train;

% Config data structure====================================================
config.OCSVM        = true;        % if true, then use OCSVM instead of Chi-square detector
config.adptQ        = false;        % if true, then adaptively estimate process noise covariance matrix Q
config.adptR        = false;        % if true, then adaptively estimate measurement noise covariance matrix R
config.use_CF       = true;         % true if using CF model
config.detection    = true;        % true if start using fault detecter
config.use_predict  = false;        % true if replacing estimate as predict when anomaly detected
config.print        = 1000;         % interval of iterations for progress printing
config.ukf          = false;        % true if using Unscented Kalman Filter      
config.bias_correct = false;        % true if enable bias correction in EKF

if(config.ukf)                      % UKF parameters
    config.alpha    = 1e-3;
    config.ki       = 0;
    config.beta     = 2;
end
config.OCSVM_threshold  = [0.8; 1; 3];        % OCSVM model threshold for training
config.R                = diag([0.01,0.01]);    % observation noise covariance

if(config.bias_correct)
    config.Q                = diag([0.5,0.3,1e2]);  %diag([0.5,0.3]);% process noise covariance
    config.H                = [1,0,1;0,1,0];    % observation matrix
else
    config.Q                = diag([0.5,0.3]);  % process noise covariance
    config.H                = [1,0;0,1];        % observation matrix
end
config.r                = 0.8;                  % Chi-square detector parameter
config.delta_t          = 0.1;                  % sensor sampling time interval in seconds
config.tau              = 0.0;                  % time delay
config.N_ocsvm          = 10;                   % Time window length for OCSVM
config.N                = 2;                    % time window length for AdEKF

config.plot             = true;                 % true if generate plots

weight_vector = [3,7];                          % fogeting factor for adaptive EKF
config.weight = weight_vector./sum(weight_vector);

% IDM CF model parameter===================================================
idm_para.a = 0.73;      % maximum acceleration
idm_para.b = 1.67;      % comfortable deceleration
idm_para.sigma = 4;     % acceleration exponent 
idm_para.s0 = 2;        % minimum distance (m)
idm_para.T = 1.0;       % safe time headway (s)
idm_para.v0 = 24;       % desired velocity (m/s)
idm_para.a_max = -0.2;   % max acceleration of random term 
idm_para.a_min = -0.4;  % max deceleration of random term
idm_para.Length = 0;    % vehicle length (m)
idm_para.tau_var = 0;    % variance of random time delay
%==========================================================================
%   AnomalyConfig: 
%       .index: index of anomaly
%       .percent: threshold of anomaly occurance in scale [0,1]
%       .anomaly_type: list of anomaly types, should be a list in choice of
%       'Instant','Bias','Drift'
%       .dur_length: the max duration of anomaly, generated as uniform
%                    distribution. The overall percentage of anomaly is
%                    .durlength x .percent /2
%       .NoiseVar: Noise type anomaly standard covariance matrix with dimension m x m
%       .BiasVar: Bias type anomaly covariance matrix with dimension m x m
%       .DriftVar: Drift type anomaly max value

AnomalyConfig.percent       = 0.005;
AnomalyConfig.anomaly_type  = {'Noise','Bias','Drift'};
AnomalyConfig.dur_length    = 20;
AnomalyConfig.NoiseVar      = diag(sqrt([5, 5]));
AnomalyConfig.BiasVar       = diag(sqrt([5, 5]));
AnomalyConfig.DriftMax      = [5, 5];
AnomalyConfig.seed          = 1; % random seed controler
%% Generate baseline data
[x_l,v_l]           = data_process(raw_data);   % get leading vehicle location x_l, speed v_l and acceleration a_l for training 
[x_l_test,v_l_test] = data_process(s);          % get leading vehicle location x_l, speed v_l and acceleration a_l for testing
% Generate following vehicle location x_f, speed v_f and acceleration a_l based on a
% car-following model 

x0 = 5;    % initial location of following vehicle
v0 = 1;     % initial speed of following vehicle

tau     = config.tau;       % human/sensor reaction time delay with unit "s"
delta_t = config.delta_t;   % sampling time interval with unit "s"

t  = floor(tau/delta_t);    % time delay in discrete state-transition model

s_f_train   = cf_model(x_l,v_l,x0,v0,delta_t,t,tau,idm_para);

s_f         = cf_model(x_l_test,v_l_test,x0,v0,delta_t,t,tau,idm_para);

save(strcat(filePath, 'following_state_test_baseline.mat'),'s_f')         % testing data
writematrix(s_f, strcat(filePath,'following_state_test_baseline.csv'))
save(strcat(filePath, 'following_state_train_baseline.mat'), 's_f_train') 
writematrix(s_f_train, strcat(filePath,'following_state_train_baseline.csv'))
% training data
%% Run experiments
s   = s(1:end,:)';
s_f = s_f(1:end,:)'; % baseline of testing data

s_train     = s_train(1:end,:)';
s_f_train   = s_f_train(1:end,:)';
% Generate anomalous data
[s_la, s_fa, AnomalyConfig] = generateAnomaly(s, s_f, AnomalyConfig); 
AnomalyIdx = AnomalyConfig.index; % ground truth

s_test = s_la; s_f_test = s_fa; % test dataset
writematrix(s_fa', strcat(filePath,'following_state_test_anomalous.csv'))
writematrix(s_test', strcat(filePath,'leading_state_test_anomalous.csv'))
writematrix(AnomalyIdx', strcat(filePath,'anomaly_index.csv'))
%% Run Models
% Generate statistics for baseline data
if(config.OCSVM)
    fprintf('Entering training phase...\n');
    config.OCSVM = false;
    [~,~,p0] = CfFilter(s_train, s_f_train, config, idm_para, s_f_train); 
    config.OCSVM = true;
elseif(config.plot)
    fprintf('Entering training phase...\n');
    [~,~,p0] = CfFilter(s_train, s_f_train, config, idm_para, s_f_train); 
end
    
% Train several OCSVM models with different sensitivity levels
if(config.OCSVM)
    [SVMModel1,SVMModel2,SVMModel3,SVMModel4] = trainmodel(p0.innov,config.OCSVM_threshold);
    
    config.SVMModel1 = SVMModel1;
    config.SVMModel2 = SVMModel2;
    config.SVMModel3 = SVMModel3;
    config.SVMModel4 = SVMModel4;
    
 % Test OCSVM
    fprintf('Entering testing phase...\n');
    [shat,err,p]    = CfFilter(s_test, s_f_test, config, idm_para, s_f);    
    err             = logical(err');
    s               = s_test';
    s_f             = s_f_test';
% Test chi^2 detector   
else
    fprintf('Entering testing phase...\n');
    [shat,err,p]    = CfFilter(s_test, s_f_test, config, idm_para, s_f);
    err             = logical(err');
end

%% Generate summary

anomaly_idx = AnomalyConfig.index(1,:) | AnomalyConfig.index(2,:);

TP     = nnz(err(anomaly_idx==1));  % true positive
FP     = nnz(err(anomaly_idx==0));  % false positive
TN     = nnz(~err(anomaly_idx==0)); % true negative
FN     = nnz(~err(anomaly_idx==1)); % false negative

f1   = 2*TP/(2*TP+FP+FN);
acc  = (TP+TN)/(TN+FN+FP+TP);
spec = TN/(TN+FP);
sen  = TP / (TP + FN);
ppv  = TP/(TP+FP);
fpRate = 1-spec;

metric_name     = {'TP';'FP';'TN';'FN';'F1';'Accuracy';'Specificity';'Sensitivity'...
    ;'Precision';'FP rate'};
metric_values   = [TP;FP;TN;FN;f1;acc;spec;sen;ppv;fpRate];

Summary.configuration       = config;
Summary.car_following_para  = idm_para;
Summary.anomaly             = AnomalyConfig;
Summary.results             = table(metric_name,metric_values);

filename = 'Summary.mat';
save(strcat(filePath,filename),'Summary')

disp(Summary.results)

%% Plotting
if(config.plot)
    close all;
    
    s_cf = s_f;
    
    if(config.bias_correct)
        shat0 = shat;
        shat= config.H*shat0;
    end
    
    figure(1)
    
    subplot(211)
    plot(1:length(s_cf(1,:)),s_cf(1,:)); 
    hold on, plot(1:length(s_f_test(1,:)),s_f_test(1,:)); 
    plot(1:length(shat(1,:)),shat(1,:)); hold off
    legend('CF simulated','real data','filtered data')
    title('following vehicle location')
    
    rmse_loc = sqrt(mean((s_f_test(1,:) - shat(1,:)).^2))
    
    subplot(212)
    plot(1:length(s_cf(2,:)),s_cf(2,:)); 
    hold on, plot(1:length(s_f_test(2,:)),s_f_test(2,:));
    plot(1:length(shat(2,:)),shat(2,:)); hold off
    legend('CF simulated', 'real data','filtered data')
    title('following vehicle speed')
    
    rmse_speed = sqrt(mean((s_f_test(2,:) - shat(2,:)).^2))
    
    err = immse(s_f_test,shat)
    
    [m,n] = size(s_fa);

    for i = 1:n
        s_fa(:,i) = s_fa(:,i) + config.R*randn(m,1);
    end

    xx = 1:size(AnomalyConfig.index, 2);

    x_l = s_la(1,:);
    v_l = s_la(2,:);

    x_f1 = s_fa(1,:);
    v_f1 = s_fa(2,:);
    
    figure(2)
    
    subplot(411),
    plot(v_l);hold on; plot(v_f1);legend('leading-raw','following-raw'); ylim([0,40]);
    subplot(412),
    plot(x_l);hold on; plot(x_f1);legend('leading-raw','following-raw');

    x_f2 = shat(1,:);
    v_f2 = shat(2,:);

    subplot(413),
    plot(v_l);hold on; plot(v_f2);
    % plot(xx(err),v_f2(err),'b*');
    plot(xx(AnomalyConfig.index(2,:)),v_f2(AnomalyConfig.index(2,:)),'d');
    legend('leading-raw','following-filtered','Anomaly'); ylim([0,40]);

    subplot(414),
    plot(x_l);hold on; plot(x_f2);
    plot(xx(AnomalyConfig.index(1,:)),x_f2(AnomalyConfig.index(1,:)),'d');
    legend('leading-raw','following-filtered','Anomaly');

    figure (3)
    subplot(511),
    h_x = histogram(p0.innov(1,:));
    h_x.BinWidth = 0.05;
    hold on
    mean_location = mean(p0.innov(1,:))
    h_v = histogram(p0.innov(2,:)); legend('vehicle location','vehicle speed');
    hold off
    h_v.BinWidth = 0.05;
    xlim([-5,5]);
    mean_speed = mean(p0.innov(2,:))

    subplot(512),
    hist = vecnorm(p0.innov(:,:),1);
    h_hist = histogram(hist);legend('Histogram $l1$ norm of innovation sequence','Interpreter','latex')
    h_hist.BinWidth = 0.05;
    xlim([-0,5]);

    subplot(513),
    text = '$l1$ norm of innovation sequence';
    plot(hist), legend(text,'Interpreter','latex')
    ylim([0,5]);

    subplot(514),
    text = '$\chi^2$ test statistcs sequence';
    plot(p.chi),  ylim([0,min(2*config.r,10)]), legend(text, 'Interpreter','latex');

    subplot(515),
    plot(p.rmse), legend('RMSE sequence');
    
    figure(4)
    R = sqrt(config.r);
    theta=0:0.01:2*pi;
    x=R*sin(theta);
    y=R*cos(theta);
    plot(x,y,'LineWidth',2)
    axis equal
    hold on
    
    scatter(p0.innov(1,:),p0.innov(2,:)),hold on
    scatter(mean_location,mean_speed,'filled','MarkerEdgeColor','k','MarkerFaceColor','k')
    xlim([-2.5 2.5]), ylim([-2.5 2.5])
    xlabel('Innovation of location','FontSize',16), ylabel('Innovation of speed','FontSize',16)
    grid on
    title('Scatter plot of innovation sequence','FontSize',16)
    legend('\chi^2 detector threshold','innovation','innovation centroid','FontSize',14)
    
    figure(5)
    subplot(211),
    plot(v_l,'LineWidth',3);hold on; plot(v_f1,'LineWidth',3,'LineStyle','-.');legend('leading-raw','following-raw','Location','northwest','FontSize',14); ylim([0,40]);
    
    xlabel('Time epoch (\times 100ms)','FontSize',16), ylabel('Speed (m/s)','FontSize',16)
    grid on
    
    subplot(212),
    plot(x_l,'LineWidth',3);hold on; plot(x_f1,'LineWidth',3,'LineStyle','-.');legend('leading-raw','following-raw','Location','northwest','FontSize',14);
    
    xlabel('Time epoch (\times 100ms)','FontSize',16), ylabel('Distance (m)','FontSize',16)
    grid on
    x_f2 = shat(1,:);
    v_f2 = shat(2,:);
    
    
    
    
end