% Version 2: ring configuration of platoon.

clear

dataPath = 'dataset\'; % dataset location
s = load(strcat(dataPath,'testdata.mat')); % info of the leading vehicle = s for testing n_sample * m
s_train = load(strcat(dataPath,'rawdata.mat')); % info of the leading vechicle = s_train for training n_sample * m

v_train = 10 * ones(1, 4000);
v_train(1, 100) = v_train(1, 100) - 0.5;
x_train = 10 * [0 : 0.1 : 3999 * 0.1];
x_train(101 : -1) = x_train(101 : -1) - 0.5 * 0.1;
s_train.s_train = [x_train; v_train]';

v_test = 10 * ones(1, 2000);
x_test = 10 * [0 : 0.1 : 1999 * 0.1];
s.s = [x_test; v_test]';

% Config data structure====================================================
config.OCSVM        = true;        % if true, then use OCSVM instead of Chi-square detector
config.adptQ        = false;        % if true, then adaptively estimate process noise covariance matrix Q
config.adptR        = false;        % if true, then adaptively estimate measurement noise covariance matrix R
config.use_CF       = true;         % true if using CF model
config.detection    = true;        % true if start using fault detecter
config.use_predict  = false;        % true if replacing estimate as predict when anomaly detected
config.print        = 1000;         % interval of iterations for progress printing
config.ukf          = false;        % true if using Unscented Kalman Filter
config.bias_correct = true;        % true if enable bias correction in EKF

if(config.ukf)                      % UKF parameters
    config.alpha    = 1e-3;
    config.ki       = 0;
    config.beta     = 2;
end
config.OCSVM_threshold  = [0.1;];        % OCSVM model threshold for training
config.R                = diag([0.01,0.01]);    % observation noise covariance

if(config.bias_correct)
    config.Q                = diag([0.5,0.3,1e-2]);  %diag([0.5,0.3]);% process noise covariance
    config.H                = [1,0,1;0,1,0];    % observation matrix
else
    config.Q                = diag([0.5,0.3]);  % process noise covariance
    config.H                = [1,0;0,1];        % observation matrix
end
config.r                = 0.3;                  % Chi-square detector parameter
config.delta_t          = 0.1;                  % sensor sampling time interval in seconds
config.tau              = 0.5;                  % time delay
config.N_ocsvm          = 10;                   % Time window length for OCSVM
config.N                = 2;                    % time window length for AdEKF

config.plot             = true;                 % true if generate plots

weight_vector = [3,7];                          % fogeting factor for adaptive EKF
config.weight = weight_vector./sum(weight_vector);

% IDM CF model parameter===================================================
idm_para.a = 1;      % maximum acceleration
idm_para.b = 2;      % comfortable deceleration
idm_para.sigma = 4;     % acceleration exponent
idm_para.s0 = 2;        % minimum distance (m)
idm_para.T = 1.1;       % safe time headway (s)
idm_para.v0 = 33.33;       % desired velocity (m/s)
idm_para.a_max = -0.2;   % max acceleration of random term
idm_para.a_min = -0.4;  % max deceleration of random term
idm_para.Length = 5;    % vehicle length (m)
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
AnomalyConfig.NoiseVar      = diag(sqrt([0.5, 0.2]));
AnomalyConfig.BiasVar       = diag(sqrt([0.5, 0.2]));
AnomalyConfig.DriftMax      = [0.5, 0.2];
AnomalyConfig.seed          = 1; % random seed controler

%==========================================================================
%   PlatoonConfig:
PlatoonConfig.N_platoon = 10;
% PlatoonConfig.alpha = [1];
PlatoonConfig.alpha = [0.7, 0.2, 0.1];
PlatoonConfig.headway = 30;
PlatoonConfig.v_init = eq_h(idm_para, PlatoonConfig.headway);
PlatoonConfig.perturbation = true;
PlatoonConfig.attack_id = 3;

N_train = 4000;
N_test = 2000;
% platoon trajectory for training
[X_train, V_train, X_syn_train, V_syn_train] = platoon_model_v2(config, idm_para, ...
    PlatoonConfig, N_train);

% platoon trajectory for testing
PlatoonConfig.inject_anomaly = true;
[X_test, V_test, X_syn_test, V_syn_test] = platoon_model_v2(config, idm_para, ...
    PlatoonConfig, N_test);

% headway of each following vehicle in platoon
N_platoon = PlatoonConfig.N_platoon;
E_train = zeros(N_platoon, size(X_train, 2));
E_test = zeros(N_platoon, size(X_test, 2));

headway = PlatoonConfig.headway;
Length = idm_para.Length;
for i = 1 : N_platoon
    if i == 1
        E_train(i, :) = X_train(N_platoon, :) - X_train(1, :) + X_train(1, 1) + (headway + Length);
        E_test(i, :) = X_test(N_platoon, :) - X_test(1, :) + X_test(1, 1) + (headway + Length);
    else
        E_train(i, :) = X_train(i - 1, :) - X_train(i, :);
        E_test(i, :) = X_test(i - 1, :) - X_test(i, :);
    end
end

max_E_train = max(abs(E_train(1:N_platoon, :)-headway-idm_para.Length), [], 2);
max_E_test = max(abs(E_test(1:N_platoon, :)-headway-idm_para.Length), [], 2);


%% Inject anomalies into testing dataset
% Generate anomalous data


%% plotting
close all
figure(1)
subplot(211)
plot(0:N_platoon-1, max_E_train, ":d", "LineWidth", 1)
ylim([0, max(max_E_train)])
xlabel("Vehicle ID")
ylabel("max |e_i(t)| (m/s)")
title("Maximum spacing error in training dataset")
grid on

subplot(212)
plot(0:N_platoon-1, max_E_test, ":d", "LineWidth", 1)
ylim([0, max(max_E_test)])
xlabel("Vehicle ID")
ylabel("max|e_i(t)| (m/s)")
title("Maximum spacing error in testing dataset")
grid on

figure(2)
subplot(211)
plot(1:size(X_test, 2), X_test, "LineWidth",1)
xlabel("Time epoch (0.1 sec)")
ylabel("Location (m)")
ylim([min(X_test, [], "all"), max(X_test, [], "all")])
title("Location of vehicles in platoon of testing dataset")
grid on

subplot(212)
plot(1:size(V_test, 2), V_test, "LineWidth",1)
xlabel("Time epoch (0.1 sec)")
ylabel("Velocity (m/s)")
ylim([min(V_test, [], "all"), max(V_test, [], "all")])
title("Velocity of vehicles in platoon of testing dataset")
grid on

legendStrings = "Vehicle " + string([0:N_platoon-1]);
legend(legendStrings)

figure(3)
plot(round(E_test', 8) - headway - idm_para.Length, "LineWidth", 1)
xlabel("Time epoch (0.1 sec)")
ylabel("Spacing error e_i(t) (m/s)")
title("Spacing error over time of testing dataset")
legendStrings = "Vehicle " + string([0:N_platoon-1]);
legend(legendStrings)
grid on