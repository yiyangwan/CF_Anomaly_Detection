function [s_f_1, s_l_2, s_f_2] = platoon_model_3(x_l_1,v_l_1,delta_t,t,tau,idm_para)
% PLATOON_MODEL Summary of this function goes here
% cf_model function produces the following vehicle infomation based on the
% input of the following vehicle's infomation
% input:
%       leading vehicle location x_l, speed v_l, initial location x0 and speed v0 of the following vehicle
%       t = floor(tau/delta_t)
%       delta_t: sampling time interval
%       tau: time delay
% output:
%       location x_f, speed v_f and acceleration a_f of the following
%       vehicle
%============generate data in IDM car-following model======================
% first following vehicle
x0_1 = 10; % initial location of the first following vehicle
v0_1 = 0; % initial speed of following vehicle
s_f_1 = cf_model(x_l_1,v_l_1,x0_1,v0_1,delta_t,t,tau,idm_para);
x_f_1 = s_f_1(:,1);
v_f_1 = s_f_1(:,2);

% second following vehicle
x0_2 = 0; % initial location of the first following vehicle
v0_2 = 0; % initial speed of following vehicle
a1 = 1;
a2 = 0;
x_l_2 = a1 * x_f_1 + a2 * x_l_1;
v_l_2 = a1 * v_f_1 + a2 * v_l_1;
s_l_2 = [x_l_2, v_l_2];

s_f_2 = cf_model(x_l_2,v_l_2,x0_2,v0_2,delta_t,t,tau,idm_para);
end

