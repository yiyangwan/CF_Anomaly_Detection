function s = idm(x_l,v_l,x_f_pre,v_f_pre,delta_t,t,tau,idm_para)
% This function uses IDM car-following model which generate the location,
% speed, and acceleration of the following vehicle.
% delta_t: sampling interval
% t: number of samples caused by time delay
% tau: time delay in seconds

s = [x_f_pre;v_f_pre]; % define state vector
u = [x_l;v_l]; % define input vector

s = g_sd(s,u,delta_t,t,tau,idm_para);
% x_f = s(1);
% v_f = s(2);
end

function s1 = g_sd(s,u,delta_t,t,tau,idm_para)
xf = s(1);
vf = s(2);

xl = u(1);
vl = u(2);

% generate acceleration during time delay as
% a normal distributed random variable within the range [a_min, a_max]
% a_max = 0.73; a_min = -9.65;
% a_max = idm_para.a_max; a_min = idm_para.a_min;
% pd = makedist('Normal', 'mu', (a_max+a_min)/2, 'sigma', sqrt(a_max - a_min));
% tpd = truncate(pd,a_min,a_max);
% a_random = random(tpd);


% IDM CF model in discrete form with time delay
% parameter: a, b, sigma, v0, T, s0

a = idm_para.a; % maximum acceleration
b = idm_para.b; % comfortable deceleration
sigma = idm_para.sigma; % acceleration exponent
s0 = idm_para.s0; % minimum distance (m)
T = idm_para.T; % safe time headway (s)
v0 = idm_para.v0; % desired velocity (m/s)
Length = idm_para.Length; % vehicle length
% v_1 = delta_t * a*(1 - (vf/v0)^sigma - (distance(vf,vf-vl,a,b,T,s0)/(xl-xf))^2) + vf+a_random*t*delta_t;
accel = a*(1 - (vf/v0)^sigma - (distance(vf,vf-vl,a,b,T,s0)/(xl-xf-Length))^2);
v_f1 = 0.5* delta_t * accel + vf;
% x_f1 = vf * delta_t + xf + a_random * tau * delta_t ;
% x_f1 = vf * delta_t + xf + (-0.3) * tau * delta_t ;
x_f1 = v_f1 * delta_t + xf;
s1 = [x_f1; v_f1];
end

function s = distance(vf,delta_v,a,b,T,s0)
s = s0 + vf*T + vf*delta_v/(2*sqrt(a*b));
end