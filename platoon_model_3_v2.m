function [s, s_train] = platoon_model_3_v2(config,idm_para)
dataPath = 'dataset\'; % dataset location
s = load(strcat(dataPath,'testdata.mat')); % info of the leading vehicle = s for testing n_sample * m
s_train = load(strcat(dataPath,'rawdata.mat')); % info of the leading vechicle = s_train for training n_sample * m

%% Generate baseline data
[x_l_train,v_l_train] = data_process(s_train.s_train);   % get leading vehicle location x_l, speed v_l and acceleration a_l for training
[x_l_test,v_l_test] = data_process(s.s);          % get leading vehicle location x_l, speed v_l and acceleration a_l for testing

if x_l_train(1) < 20
    x_l_train = x_l_train + 20 - x_l_train(1);
end

if x_l_test(1) < 20
    x_l_test = x_l_test + 20 - x_l_test(1);
end
% Generate following vehicle location x_f, speed v_f and acceleration a_l based on a
% car-following model

tau = config.tau; % human/sensor reaction time delay with unit "s"
delta_t = config.delta_t; % sampling time interval with unit "s"
t  = ceil(tau/delta_t); % time delay in discrete state-transition model

% first following vehicle
x0_1 = 10; % initial location of the first following vehicle
v0_1 = 0; % initial speed of following vehicle
s_f_1_train = cf_model(x_l_train,v_l_train,x0_1,v0_1,delta_t,t,tau,idm_para);
x_f_1_train = s_f_1_train(:,1);
v_f_1_train = s_f_1_train(:,2);

s_f_1_test = cf_model(x_l_test,v_l_test,x0_1,v0_1,delta_t,t,tau,idm_para);
x_f_1_test = s_f_1_test(:,1);
v_f_1_test = s_f_1_test(:,2);

% second following vehicle
x0_2 = 0; % initial location of the first following vehicle
v0_2 = 0; % initial speed of following vehicle
a1 = 0.8;
a2 = 1-a1;
x_l_2_train = a1 * x_f_1_train + a2 * x_l_train;
v_l_2_train = a1 * v_f_1_train + a2 * v_l_train;

s_f_2_train = cf_model(x_l_2_train,v_l_2_train,x0_2,v0_2,delta_t,t,tau,idm_para);
x_f_2_train = s_f_2_train(:,1);
v_f_2_train = s_f_2_train(:,2);
s_train = [x_l_2_train, v_l_2_train];

x_l_2_test = a1 * x_f_1_test + a2 * x_l_test;
v_l_2_test = a1 * v_f_1_test + a2 * v_l_test;

s_f_2_test = cf_model(x_l_2_test,v_l_2_test,x0_2,v0_2,delta_t,t,tau,idm_para);
x_f_2_test = s_f_2_test(:,1);
v_f_2_test = s_f_2_test(:,2);
s = [x_l_2_test, v_l_2_test];
end

