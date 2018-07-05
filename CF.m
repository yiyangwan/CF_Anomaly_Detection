clear
close all
load('rawdata.mat')
raw_data   = s;
[x_l,v_l] = data_process(raw_data); % get leading vehicle location x_l, speed v_l and acceleration a_l

% Generate following vehicle location x_f, speed v_f and acceleration a_l based on a
% car-following model 

x0 = 10; % initial location of following vehicle
v0 = 1; % initial speed of following vehicle

% IDM CF model parameter===================================================
idm_para.a = 0.73; % maximum acceleration
idm_para.b = 1.67; % comfortable deceleration
idm_para.sigma = 4; % acceleration exponent 
idm_para.s0 = 2; % minimum distance (m)
idm_para.T = 1.5; % safe time headway (s)
idm_para.v0 = 24; % desired velocity (m/s)
idm_para.a_max = 0.1; % max acceleration of random term 
idm_para.a_min = -0.1; % max deceleration of random term

config.OCSVM = true; % if true, then use OCSVM instead of Chi-square detector
config.OCSVM_threshold = [3; 4.5; 6]; % OCSVM model threshold for training
config.R = diag([1,0.5]); % observation noise covariance
config.Q = diag([0.8,0.05]); % process noise covariance
config.H = eye(2); % observation matrix
config.N = 15; % Time window length for AdEKF
config.r = 5; % Chi-square detector parameter
config.delta_t = 0.1; % sensor sampling time interval in seconds
config.tau = 0.01; % time delay
config.N_ocsvm = 15; % Time window length for OCSVM

tau = config.tau; % human/sensor reaction time delay with unit "s"
delta_t = config.delta_t; % sampling time interval with unit "s"
t  = ceil(tau/delta_t); % time delay in discrete state-transition model

s_f = cf_model(x_l,v_l,x0,v0,delta_t,t,tau,idm_para);
x_f = s_f(:,1);
v_f = s_f(:,2);
save('following_state.mat','s_f')

%% plot
subplot(211)
plot(v_l);hold on; plot(v_f);legend('leading','following');
subplot(212)
plot(x_l);hold on; plot(x_f);legend('leading','following');
