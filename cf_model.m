%% car following model function
function s_f = cf_model(x_l,v_l,x0,v0,delta_t,t,tau,idm_para)
% cf_model function produces the following vehicle infomation based on the
% input of the following vehicle's infomation
% input: 
%       leading vehicle location x_l, speed v_l, initial location x0 and speed v0 of the following vehicle
% output: 
%       location x_f, speed v_f and acceleration a_f of the following
%       vehicle

%============generate data in IDM car-following model======================
n_step = length(x_l);

x_f = zeros(n_step,1);
v_f = zeros(n_step,1);

x_f_pre = x0; 
v_f_pre = v0;

if (t>=0)
    x_f(1:t) = x0;
    v_f(1:t) = v0;
end
for i = 1:n_step
    s = idm(x_l(i),v_l(i),x_f_pre,v_f_pre,delta_t,t,tau,idm_para);
    x_f(i+t) = s(1) ; v_f(i+t) = s(2); 
    x_f_pre = x_f(i+t);
    v_f_pre = v_f(i+t);
end
s_f = [x_f,v_f];
end