function s = cidm(delta_s, delta_v,xf,vf,delta_t,t,tau,idm_para)

a = idm_para.a; % maximum acceleration
b = idm_para.b; % comfortable deceleration
sigma = idm_para.sigma; % acceleration exponent
s0 = idm_para.s0; % minimum distance (m)
T = idm_para.T; % safe time headway (s)
v0 = idm_para.v0; % desired velocity (m/s)
% Length = idm_para.Length; % vehicle length

accel = a*(1 - (vf/v0)^sigma - ...
    (distance(vf,delta_v,a,b,T,s0)/(delta_s))^2);
% if delta_s < 0 || vf < 0
%     disp(accel)
%     error("negative velocity!")
% end
vf1 = delta_t * accel + vf;
xf1 = vf1 * delta_t + xf;
s = [xf1; vf1];
end

function s = distance(vf,delta_v,a,b,T,s0)
s = s0 + vf*T + vf*delta_v/(2*sqrt(a*b));
end