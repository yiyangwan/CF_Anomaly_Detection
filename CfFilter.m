function [shat,error_idx,p] = CfFilter(s_l, s_f ,config, idm_para, groundtruth)
% The function CfFilter will filter signal s_f given the input
% signal s_l. Also wil do the detection. Anomaly can come from both s_l and s_f.
% 
% Input: s_l: sensor reading of the leading vehilce's state with dimension m x n_sample
%        s_f: sensor reading of the following vehicle's state as the
%        baseline with dimension m x n_sample
%        R : measurement noise covariance matrix with dimension m x m
%        idm_para: structure of IDM CF model parameters
%        groundtruth: ground truth with dimension m x m
%
% Output: shat: filtered sensor reading of the following vehicle's state
%         error: index of anomaly
%         p: p.chi: chi-square statistics 
%            p.innov: normalized innovation sequence

n_sample    = max(size(s_f));
m           = size(s_f,1);      % dimension of state

shat        = zeros(m,n_sample);
p.chi       = zeros(1,n_sample); % chi-square statistics 
p.innov  	= zeros(m,n_sample); % normalized innovation sequences
p.score     = zeros(1,n_sample); % score of OCSVM classifier
error_idx   = zeros(n_sample,1);

delta_t = config.delta_t;   % sensor sampling time interval
s0      = s_f(:,1);         % initial state prediction for EKF

P_hat   = diag(ones(2,1)); 	% initial state prediction covariance for EKF
if(~config.ukf)
    x_hat   = s0;
    
else
    c       = sqrt(config.alpha^2*(m+config.ki));
    x_hat.x = s0;
    x_hat.X = sigmas(x_hat.x,P_hat,c);
end

N       = config.N;         % Time window length for Adaptive EKF
N_ocsvm = config.N_ocsvm;   % Time window length for OCSVM

Ccum    = diag(zeros(m,1)); % initial weighted sum of outer product of innovation
Ucum    = diag(zeros(m,1)); % initial estimated measurement noises covariance matrix
vlist   = zeros(m,m,N);     % stack for innovation sequence with length N
ulist   = zeros(m,m,N);     % stack for residual sequence with length N
psum    = zeros(m,1);       % initial sum of N_ocsvm-1 normalized innovation

p.rmse  = zeros(m,1);       % stack to store RMSE vector

s_f     = s_f + config.R*randn(m,n_sample); % add noise into measurement

H       = config.H; % measurement matrix

if(config.use_CF)
    f       = @(tt,ss,uu,ZZ) CF_idm(tt,ss,uu,ZZ,idm_para);  % function handle for motion model
    del_f   = @(ss,uu) CF_idm_der(ss,uu,idm_para);          % function handle for the Jacobian of motion model

    CF      = @(s,u) CF_idm2(s,u,idm_para);

else
    f       = @(tt,ss,uu,ZZ) non_CF(tt,ss,ZZ);  % function handle for motion model without considering CF model
    del_f   = @(ss,uu) non_CF_der(ss);          % function handle for the Jacobian of motion model without considering CF model

end

h       = @(s) H*s; % function handle for measurement model
del_h   = @(s) H;   % function handle for the Jacobian of motion model


for i = 1:n_sample
    if(rem(i,config.print) == 0)
        fprintf('processing %d th samples...\n',i);
    end
    
    if(~config.ukf)
        [x_next,P_next,x_dgr,P_dgr,p1,K,error1] = ekf(f,h,s_f(:,i),del_f,del_h,x_hat,P_hat,s_l(:,i),groundtruth(:,i),CF,@dde_his,(i-1)*delta_t,i*delta_t,config,psum);
    else
        [x_next,P_next,x_dgr,P_dgr,p1,K,error1] = ukf(f,h,s_f(:,i),x_hat,P_hat,s_l(:,i),groundtruth(:,i),CF,@dde_his,(i-1)*delta_t,i*delta_t,config,psum);    
    end
    vlist(:,:,1:N-1)    = vlist(:,:,2:end);         %shift left
    vlist(:,:,N)        = p1.y_tilde*p1.y_tilde';
    
    ulist(:,:,1:N-1)    = ulist(:,:,2:end);         %shift left
    ulist(:,:,N)        = p1.residual*p1.residual';
    
    p.rmse(i) = p1.RMSE;
    % Adaptively estimate covariance matrices Q and R based on innovation
    % and residual sequences
    for j = 1:N
       Ccum = Ccum + config.weight(j) * (K*squeeze(vlist(:,:,j))*K');
       Ucum = Ucum + config.weight(j) * ( squeeze(ulist(:,:,j)) + config.H * P_next * config.H' );    
    end
    
    if(config.adptQ)
        config.Q = Ccum; % compute adaptively Q based on innovation sequence
    end
    if(config.adptR)
        config.R = Ucum; % compute adaptively R based on residual sequence
    end
    
    Ccum = diag(zeros(m,1)); % reset sum every loop
    Ucum = diag(zeros(m,1)); % reset sum every loop
    
    shat(:,i)       = x_dgr;
    error_idx(i)    = error1;
    p.chi(i)        = p1.chi;
    p.innov(:,i)    = p1.innov;
    
    if(config.OCSVM)
        p.score(i)  = p1.score;
    end
    
    if(i>=N_ocsvm-1)
        psum = sum(p.innov(:,i-N_ocsvm+2:i),2); 
    else
        psum = sum(p.innov(:,1:i),2);
    end
    
    x_hat = x_next;
    P_hat = P_next;
    
    % Ignore warining 
    
    [~, MSGID] = lastwarn();
    if(~isempty(MSGID))
        warning('off', MSGID);
    end
end

end

function s_his = CF_his(t,x_hat)
% History function for CF_idm.
s_his = x_hat;

end

function P_his = P_his(t,P_hat)
% History function for f_del1.
P_his = P_hat;

end

function s_d = CF_idm(t,s,u,Z,idm_para)
% This function inplements IDM CF model with time delay tau.
%   The differential equations
%
%        s'_1(t) = s_2(t-0.1)
%        s'_2(t) = a*(1-(s_2(t-0.1)/v0)^sigma - ((s0 + s_2(t-0.1)*T + s_2(t-0.1)*(s_2(t-0.1) - u_2(t-0.1))/(2*sqrt(a*b))) )
%   are created on [0, 200] with history s_1(t) = 0, s_2(t) = 0 for
%   t <= 0.

% Parameters---------------------------------------------------------------
% a_max = idm_para.a;
a       = idm_para.a;       % maximum acceleration
b       = idm_para.b;       % comfortable deceleration
sigma   = idm_para.sigma;   % acceleration exponent 
s0      = idm_para.s0;      % minimum distance (m)
T       = idm_para.T;       % safe time headway (s)
v0      = idm_para.v0;      % desired velocity (m/s)

slag1   = Z(:,1);
u1      = u(1,:); % input from the leading vehicle
u2      = u(2,:); % input from the leading vehicle
s_d     = [ slag1(2);
            a*(1 - (slag1(2)/v0)^sigma - ...
            (distance(slag1(2),slag1(2)-u2,a,b,T,s0)/(u1-slag1(1)))^2)];
end

function s_d = non_CF(t,s,Z)
% This function inplements non_CF motion model with time delay.

slag1   = Z(:,1);

s_d     = [slag1(2);
       0];
end

function s_der = CF_idm_der(s,u,idm_para)
% Jacobian of motion model

% Parameters---------------------------------------------------------------
a = idm_para.a; % maximum acceleration
b = idm_para.b; % comfortable deceleration
sigma = idm_para.sigma; % acceleration exponent 
s0 = idm_para.s0; % minimum distance (m)
T = idm_para.T; % safe time headway (s)
v0 = idm_para.v0; % desired velocity (m/s)

s_der = [0, 1; a*(-2 * ( s0+s(2)*T + ( s(2)*(s(2)-u(2)) )/(2*sqrt(a*b)) )^2 * (u(1)-s(1))^(-3) ), ...
    a*( -sigma*(1/v0) * (s(2)/v0)^(sigma-1) - 2*(1/(u(1) - s(1)) )^2 * (s0 + s(2)*T+ ( s(2)*(s(2)-u(2))/2/sqrt(a*b) ) ) * (T + 1/sqrt(a*b)/2 * (2*s(2) - u(2)) ) ) ];
end

function s_der = non_CF_der(s)
% Jacobian of motion model without considering CF model

s_der = [0,1;0,0];
end

function s = distance(vf,delta_v,a,b,T,s0)
s = s0 + vf*T + vf*delta_v/(2*sqrt(a*b));
end

function s_d = CF_idm2(s,u,idm_para)
% This function inplements IDM CF model with time delay t.
%   The differential equations
%
%        s'_1(t) = s_2(t-0.1)
%        s'_2(t) = a*(1-(s_2(t-0.1)/v0)^sigma - ((s0 + s_2(t-0.1)*T + s_2(t-0.1)*(s_2(t-0.1) - u_2(t-0.1))/(2*sqrt(a*b))) )
%   are created on [0, 200] with history s_1(t) = 0, s_2(t) = 0 for
%   t <= 0.

% Parameters---------------------------------------------------------------
a       = idm_para.a;       % maximum acceleration
b       = idm_para.b;       % comfortable deceleration
sigma   = idm_para.sigma;   % acceleration exponent 
s0      = idm_para.s0;      % minimum distance (m)
T       = idm_para.T;       % safe time headway (s)
v0      = idm_para.v0;      % desired velocity (m/s)

u1      = u(1,:);           % input from the leading vehicle
u2      = u(2,:);           % input from the leading vehicle
s_d     = [ s(2);
            a*(1 - (s(2)/v0)^sigma - ...
            (distance(s(2),s(2)-u2,a,b,T,s0)/(u1-s(1)))^2)];
end

function sys_his = dde_his(t,s)
sys_his = s;
end