clear

filePath = 'dataset\'; % dataset location

% Config data structure====================================================
config.OCSVM        = false;        % if true, then use OCSVM instead of Chi-square detector
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
config.OCSVM_threshold  = [0.5; 1; 3];        % OCSVM model threshold for training
config.R                = diag([0.01,0.01]);    % observation noise covariance

if(config.bias_correct)
    config.Q                = diag([0.5,0.3,1e2]);  %diag([0.5,0.3]);% process noise covariance
    config.H                = [1,0,1;0,1,0];    % observation matrix
else
    config.Q                = diag([0.5,0.3]);  % process noise covariance
    config.H                = [1,0;0,1];        % observation matrix
end
config.r                = 0.1;                  % Chi-square detector parameter
config.delta_t          = 0.1;                  % sensor sampling time interval in seconds
config.tau              = 0.5;                  % time delay
config.N_ocsvm          = 10;                   % Time window length for OCSVM
config.N                = 2;                    % time window length for AdEKF

config.plot             = false;                 % true if generate plots

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
AnomalyConfig.NoiseVar      = diag(sqrt([1, 1]));
AnomalyConfig.BiasVar       = diag(sqrt([1, 1]));
AnomalyConfig.DriftMax      = [1, 1];
AnomalyConfig.seed          = 1; % random seed controler
%% Generate baseline data
[s, s_train] = platoon_model_3_v2(config,idm_para);
raw_data = s_train;
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

else
    % Test chi^2 detector
    fprintf('Entering testing phase...\n');
    [shat,err,p]    = CfFilter(s_test, s_f_test, config, idm_para, s_f);
    err             = logical(err');

    anomaly_idx = AnomalyConfig.index(1,:) | AnomalyConfig.index(2,:);
    sen = zeros(100, 1);
    ppv = zeros(100, 1);
    fpRate = zeros(100, 1);
    
    num_thresh = 100;
    qvals = (1:(num_thresh-1))/num_thresh;
    rr = [0 quantile(p.chi,qvals)];
    for i = 1:length(rr)
%         disp(i)
        config.r = rr(i);
        [~,err_temp,p1_temp]    = CfFilter(s_test, s_f_test, config, idm_para, s_f);
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
    auc_roc = trapz([0;flip(fpRate);1], [0;flip(sen);1])
    auc_prc = trapz([0;flip(sen);1], [1; flip(ppv); 0])
 
end


