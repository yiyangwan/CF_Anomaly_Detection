function s = gm(x_l,v_l,x_f_pre,v_f_pre,delta_t,t,tau,idm_para)
% This function uses GM car-following model which generate the location,
% speed, and acceleration of the following vehicle.

s = [x_f_pre;v_f_pre]; % define state vector
u = [x_l;v_l]; % define input vector

s = t_gm(s,u,delta_t,t,tau,idm_para);
end

function s1 = t_gm(s,u,delta_t,t,tau,idm_para)
xf = s(1);
vf = s(2);

xl = u(1);
vl = u(2);

a_max = idm_para.a_max; a_min = idm_para.a_min;
a_random = sqrt(a_max - a_min)*randn + (a_max+a_min)/2; % generate acceleration during time delay as 
% a normal distributed random variable within the range [a_min, a_max], 
% where a_min = -9.65m^2/s, a_max = 0.73m^2/s


% IDM CF model in discrete form with time delay
% parameter: a, b, sigma, v0, T, s0

a = idm_para.a; % maximum acceleration
b = idm_para.b; % comfortable deceleration
sigma = idm_para.sigma; % acceleration exponent 
s0 = idm_para.s0; % minimum distance (m)
T = idm_para.T; % safe time headway (s)
v0 = idm_para.v0; % desired velocity (m/s)

% v_1 = delta_t * a*(1 - (vf/v0)^sigma - (distance(vf,vf-vl,a,b,T,s0)/(xl-xf))^2) + vf+a_random*t*delta_t;
v_f1 = delta_t * a*(1 - (vf/v0)^sigma - (distance(vf,vf-vl,a,b,T,s0)/(xl-xf))^2) + vf;
x_f1 = vf * delta_t + xf +a_random * tau * delta_t;
s1 = [x_f1;v_f1];
end