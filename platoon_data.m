clear
close all
dataPath = 'dataset\'; % dataset location
load(strcat(dataPath,'testdata.mat')) % info of the leading vehicle = s for testing n_sample * m
load(strcat(dataPath,'rawdata.mat')) % info of the leading vechicle = s_train for training n_sample * m

raw_data   = s;
[x_l_1,v_l_1] = data_process(raw_data); % get leading vehicle location x_l, speed v_l and acceleration a_l
x_l_1 = x_l_1 + 10;
% Generate following vehicle location x_f, speed v_f and acceleration a_l based on a
% car-following model


% IDM CF model parameter===================================================
idm_para.a = 0.73; % maximum acceleration
idm_para.b = 1.67; % comfortable deceleration
idm_para.sigma = 4; % acceleration exponent
idm_para.s0 = 2; % minimum distance (m)
idm_para.T = 1.5; % safe time headway (s)
idm_para.v0 = 24; % desired velocity (m/s)
idm_para.a_max = 0.1; % max acceleration of random term
idm_para.a_min = -0.1; % max deceleration of random term
idm_para.Length = 5; % vehicle length (m)
idm_para.tau_var = 0; % variance of stochastic time delay

config.OCSVM = true; % if true, then use OCSVM instead of Chi-square detector
config.OCSVM_threshold = [3; 4.5; 6]; % OCSVM model threshold for training
config.R = diag([1,0.5]); % observation noise covariance
config.Q = diag([0.8,0.05]); % process noise covariance
config.H = eye(2); % observation matrix
config.N = 15; % Time window length for AdEKF
config.r = 5; % Chi-square detector parameter
config.delta_t = 0.1; % sensor sampling time interval in seconds
config.tau = 0.7; % time delay
config.N_ocsvm = 15; % Time window length for OCSVM

tau = config.tau; % human/sensor reaction time delay with unit "s"
delta_t = config.delta_t; % sampling time interval with unit "s"
t  = ceil(tau/delta_t); % time delay in discrete state-transition model

% first following vehicle
x0_1 = 10; % initial location of the first following vehicle
v0_1 = 0; % initial speed of following vehicle
s_f_1 = cf_model(x_l_1,v_l_1,x0_1,v0_1,delta_t,t,tau,idm_para);
x_f_1 = s_f_1(:,1);
v_f_1 = s_f_1(:,2);

% second following vehicle
x0_2 = 0; % initial location of the first following vehicle
v0_2 = 0; % initial speed of following vehicle
x_l_2 = 0.8 * x_f_1 + 0.2 * x_l_1;
v_l_2 = 0.8 * v_f_1 + 0.2 * v_l_1;

s_f_2 = cf_model(x_l_2,v_l_2,x0_2,v0_2,delta_t,t,tau,idm_para);
x_f_2 = s_f_2(:,1);
v_f_2 = s_f_2(:,2);

%% plot
subplot(211)
plot(v_l_1);hold on; plot(v_f_1); plot(v_f_2); 
legend('leading','following 1', 'following 2', 'Location','southeast'); xlabel('Time epoch ($\times$ms)', 'Interpreter','latex'), ylabel('Speed ($\times$m/s)', 'Interpreter','latex'); xlim([0,2000])
subplot(212)
plot(x_l_1);hold on; plot(x_f_1); plot(x_f_2);
legend('leading', 'following 1', 'following 2','Location','southeast');xlabel('Time epoch ($\times$ms)', 'Interpreter','latex'), ylabel('Distance ($\times$m)', 'Interpreter','latex'); xlim([0,2000])
