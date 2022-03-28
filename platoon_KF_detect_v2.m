function [X, V, X_syn, V_syn, shat, error_idx, p, AnomalyConfig] ...
    = platoon_KF_detect_v2(config, idm_para, ...
    PlatoonConfig, N_sample, AnomalyConfig)
% Version 2: Multiple attacked vehicles
%
% config: config
% idm_para: IDM model parameter
% N: number of vehicles in platoon
% N_sample: number of samples
% s_0: trajectory of platoon lead
% alpha: vector of weights, e.g. [0.7, 0.2, 0.1]
% headway: initial headway
% v_init: initial velocity
% AnomalyConfig: anomaly config

m           = size(config.H, 2);      % dimension of state
m_measure   = 2;         % dimension of measurement

N_platoon = PlatoonConfig.N_platoon;
alpha = PlatoonConfig.alpha;
v_init = PlatoonConfig.v_init;
headway = PlatoonConfig.headway;
Length = idm_para.Length;
% number of cooperative vehicles
N_coop = length(alpha);

shat        = zeros(m,N_sample);
p.chi       = zeros(1,N_sample); % chi-square statistics
p.innov  	= zeros(m_measure,N_sample); % normalized innovation sequences
p.score     = zeros(1,N_sample); % score of OCSVM classifier
error_idx   = zeros(N_sample,1);

% Generate following vehicle location x_f, speed v_f and acceleration a_l based on a
% car-following model
tau = config.tau; % human/sensor reaction time delay with unit "s"
delta_t = config.delta_t; % sampling time interval with unit "s"
t  = ceil(tau/delta_t); % time delay in discrete state-transition model

X = zeros(N_platoon, N_sample);
X(:, 1) = (headway +  Length) * [N_platoon - 1: -1 : 0]';
V = zeros(N_platoon, N_sample) + v_init;

if t > 1
    for tt = 2 : t + 1
        X(:, tt) = X(:, tt - 1) + V(:, tt - 1) * config.delta_t;
    end
end

% Add perturbation
if PlatoonConfig.perturbation
    X(1, 1 + t) = X(1, 1 + t) - 0.5 * headway;
end

X_syn = zeros(N_platoon, N_sample);
V_syn = zeros(N_platoon, N_sample);

if(~config.bias_correct)
    s0      = [X(PlatoonConfig.attack_id, 1); V(PlatoonConfig.attack_id, 1)];         % initial state prediction for EKF
else
    s0      = [X(PlatoonConfig.attack_id, 1); V(PlatoonConfig.attack_id, 1); (-0.3) * tau * delta_t];
end

P_hat   = diag(ones(m,1)); 	% initial state prediction covariance for EKF

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
Ucum    = diag(zeros(m_measure,1)); % initial estimated measurement noises covariance matrix
vlist   = zeros(m_measure,m_measure,N);     % stack for innovation sequence with length N
ulist   = zeros(m_measure,m_measure,N);     % stack for residual sequence with length N
psum    = zeros(m_measure,1);       % initial sum of N_ocsvm-1 normalized innovation

p.rmse  = zeros(m_measure,1);       % stack to store RMSE vector

H       = config.H; % measurement matrix


if(config.use_CF)
    if(~config.bias_correct)
        f       = @(tt,ss,uu,ZZ) CF_idm(tt,ss,uu,ZZ,idm_para);  % function handle for motion model
        del_f   = @(ss,uu) CF_idm_der(ss,uu,idm_para);          % function handle for the Jacobian of motion model

        CF      = @(s,u) CF_idm2(s,u,idm_para);
    else
        f       = @(tt,ss,uu,ZZ) CF_idm_bias_corrt(tt,ss,uu,ZZ,idm_para);  % function handle for motion model
        del_f   = @(ss,uu) CF_idm_der_bias_corrt(ss,uu,idm_para);          % function handle for the Jacobian of motion model

        CF      = @(s,u) CF_idm2_bias_corrt(s,u,idm_para);
    end

else
    f       = @(tt,ss,uu,ZZ) non_CF(tt,ss,ZZ);  % function handle for motion model without considering CF model
    del_f   = @(ss,uu) non_CF_der(ss);          % function handle for the Jacobian of motion model without considering CF model

    CF      = @(s,u) non_CF2(s);
end

h       = @(s) H*s; % function handle for measurement model
del_h   = @(s) H;   % function handle for the Jacobian of motion model

anomaly_sequence = zeros(2, N_sample);
if PlatoonConfig.inject_anomaly && exist("AnomalyConfig", "var")
    rng(AnomalyConfig.seed); % control random number generator so that it can produce predictable random numbers
    anomaly_type = AnomalyConfig.anomaly_type;
    num_type = numel(anomaly_type); % number of anomaly types
end
if PlatoonConfig.inject_anomaly && exist("AnomalyConfig", "var")
    AnomalyConfig.index = zeros(2, N_sample);
    for i = t + 2 : N_sample
        seed = rand(2,1);
        if(AnomalyConfig.index(1,i) ~= 1)&&(AnomalyConfig.index(2,i) ~= 1) % make sure anomaly will not overlapped
            if ((seed(1) <=AnomalyConfig.percent) || (seed(2) <=AnomalyConfig.percent))
                msk = (seed <= AnomalyConfig.percent);
                anomaly_type_idx = randi(num_type); % uniform distribution
                type = anomaly_type{anomaly_type_idx};
                dur_length = randi(AnomalyConfig.dur_length); % uniform distribution
                if (i+dur_length-1 <= N_sample)
                    AnomalyConfig.index(:,i:i+dur_length-1) = AnomalyConfig.index(:,i:i+dur_length-1)+msk;

                    switch type
                        case 'Noise'
                            anomaly_sequence(:,i:i+dur_length-1) = anomaly_sequence(:,i:i+dur_length-1) + msk.*AnomalyConfig.NoiseVar * randn(2,dur_length);
                        case 'Bias'
                            anomaly_sequence(:,i:i+dur_length-1) = anomaly_sequence(:,i:i+dur_length-1) + msk.*AnomalyConfig.BiasVar * randn(2,1);
                        case 'Drift'
                            anomaly_sequence(:,i:i+dur_length-1) = anomaly_sequence(:,i:i+dur_length-1) + (2*randi(2,2,1)-3).*msk...
                                .*[linspace(0,rand*(AnomalyConfig.DriftMax(1)),dur_length);...
                                linspace(0,rand*(AnomalyConfig.DriftMax(2)),dur_length)];
                    end
                end
            end
        end

    end
    AnomalyConfig.anomaly_sequence = anomaly_sequence;
end

s_truth = zeros(2, N_sample);
s_truth(:, 1) = [X(PlatoonConfig.attack_id, 1); V(PlatoonConfig.attack_id, 1)];

for j = t + 2 : N_sample
    if(rem(j,config.print) == 0)
        fprintf('processing %d th samples...\n',j);
    end
    for i = 1 : N_platoon
        tau1 = normrnd(tau,idm_para.tau_var);

        if i - 1 < N_coop
            X_temp = [X(i - 1 : -1 : 1, j - t - 1); X(N_platoon: -1 : N_platoon - N_coop + i, j - t - 1) + (headway +  Length) + X(1, 1)];
            X_temp1 = [X(i, j - t - 1); X_temp(1:end-1)];
            V_temp = [V(i - 1 : -1 : 1, j - t - 1); V(N_platoon: -1 : N_platoon - N_coop + i, j - t - 1)];
            V_temp1 = [V(i, j - t - 1); V_temp(1:end-1)];
        else
            X_temp = X(i - 1 : -1 : i - N_coop, j - t - 1);
            X_temp1 = [X(i, j - t - 1); X_temp(1:end-1)];
            V_temp = V(i - 1 : -1 : i - N_coop, j - t - 1);
            V_temp1 = [V(i, j - t - 1); V_temp(1:end-1)];
        end
        x_l_syn = round(alpha * (X_temp - X_temp1 - Length), 8);
        v_l_syn = round(alpha * (-V_temp + V_temp1), 8);

        s = cidm(x_l_syn,v_l_syn,X(i, j - 1),V(i, j - 1),delta_t,t,tau1,idm_para);

        if i == PlatoonConfig.attack_id
            s_truth(:, j) = s;
            if PlatoonConfig.inject_anomaly
                s = s + anomaly_sequence(:, j) ... % add anomaly into measurement;
                    + config.R*randn(m_measure,1); % add noise into measurement;
            end
            s_l = [x_l_syn; v_l_syn];

            if(~config.ukf)
                [x_next,P_next,x_est,~,p1,K,error1] = ekf(f,h,s,del_f,del_h,x_hat,P_hat,s_l,s_truth(:,j),CF,@dde_his,(i-1)*delta_t,i*delta_t,config,psum);
            elseif(config.ukf)
                [x_next,P_next,x_est,~,p1,K,error1] = ukf(f,h,s,x_hat,P_hat,s_l,s_truth(:,j),CF,@dde_his,(i-1)*delta_t,i*delta_t,config,psum);
            end

            vlist(:,:,1:N-1)    = vlist(:,:,2:end);         %shift left
            vlist(:,:,N)        = p1.y_tilde*p1.y_tilde';

            ulist(:,:,1:N-1)    = ulist(:,:,2:end);         %shift left
            ulist(:,:,N)        = p1.residual*p1.residual';

            p.rmse(i) = p1.RMSE;
            % Adaptively estimate covariance matrices Q and R based on innovation
            % and residual sequences
            for k = 1:N
                Ccum = Ccum + config.weight(k) * (K*squeeze(vlist(:,:,k))*K');
                Ucum = Ucum + config.weight(k) * ( squeeze(ulist(:,:,k)) + config.H * P_next * config.H' );
            end

            if(config.adptQ)
                config.Q = Ccum; % compute adaptively Q based on innovation sequence
            end
            if(config.adptR)
                config.R = Ucum; % compute adaptively R based on residual sequence
            end

            Ccum = diag(zeros(m,1));            % reset sum every loop
            Ucum = diag(zeros(m_measure,1));    % reset sum every loop

            shat(:,j)       = x_est;
            error_idx(j)    = error1;
            p.chi(j)        = p1.chi;
            p.innov(:,j)    = p1.innov;

            if(config.OCSVM)
                p.score(j)  = p1.score;
            end

            if(j>=N_ocsvm-1)
                psum = sum(p.innov(:,j-N_ocsvm+2:j),2);
            else
                psum = sum(p.innov(:,1:j),2);
            end

            x_hat = x_next;
            P_hat = P_next;

            % Ignore warining
            [~, MSGID] = lastwarn();
            if(~isempty(MSGID))
                warning('off', MSGID);
            end
            s = x_est;
        end

        X(i, j) = s(1); V(i, j) = s(2);

        X_syn(i, j) = x_l_syn;
        V_syn(i, j) = v_l_syn;
    end
end
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
Length  = idm_para.Length;  % vehicle length (m)

slag1   = Z(:,1);
u1      = u(1,:); % input from the leading vehicle
u2      = u(2,:); % input from the leading vehicle
s_d     = [ slag1(2);
    a*(1 - (slag1(2)/v0)^sigma - ...
    (distance(slag1(2),u2,a,b,T,s0)/(u1))^2)];
end

function s_d = non_CF(t,s,Z)
% This function inplements non_CF motion model with time delay.

slag1   = Z(:,1);

s_d     = [slag1(2);
    0];
end

function s_d = non_CF2(s)
% This function inplements non_CF motion model with time delay.

s_d     = [s(2);
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
Length  = idm_para.Length;  % vehicle length (m)

s_der = [0, 1; ...
    0, ...
    a*( -sigma*(1/v0) * (s(2)/v0)^(sigma-1)...
    - 2*(1/u(1))^2 * (s0 + s(2)*T+ ( s(2)*u(2)/2/sqrt(a*b) ) ) ...
    * (T + u(2)/sqrt(a*b)/2) ) ];
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
Length  = idm_para.Length;  % vehicle length (m)

u1      = u(1,:);           % input from the leading vehicle
u2      = u(2,:);           % input from the leading vehicle
s_d     = [ s(2);
    a*(1 - (s(2)/v0)^sigma - ...
    (distance(s(2),u2,a,b,T,s0)/(u1))^2)];
end

function sys_his = dde_his(t,s)
sys_his = s;
end

function s_der = CF_idm_der_bias_corrt(s,u,idm_para)
% Jacobian of motion model with bias correction
% State is 3 dimentional, s_der is 3 by 3 Jacobian matrix

% Parameters---------------------------------------------------------------
a = idm_para.a; % maximum acceleration
b = idm_para.b; % comfortable deceleration
sigma = idm_para.sigma; % acceleration exponent
s0 = idm_para.s0; % minimum distance (m)
T = idm_para.T; % safe time headway (s)
v0 = idm_para.v0; % desired velocity (m/s)
Length  = idm_para.Length;  % vehicle length (m)

s_der = [0, 1, 1; ...
    0, ...
    a*( -sigma*(1/v0) * (s(2)/v0)^(sigma-1)...
    - 2*(1/u(1))^2 * (s0 + s(2)*T+ ( s(2)*u(2)/2/sqrt(a*b) ) ) ...
    * (T + u(2)/sqrt(a*b)/2) ), 0;...
    0, 0, 0];
end

function s_d = CF_idm_bias_corrt(t,s,u,Z,idm_para)
% This function inplements IDM CF model with time delay tau and bias
% correction.
%   The differential equations
%
%        s'_1(t) = s_2(t-tau) + s_3(t-tau)
%        s'_2(t) = a*(1-(s_2(t-tau)/v0)^sigma - ((s0 + s_2(t-tau)*T + s_2(t-tau)*(s_2(t-tau) - u_2(t-tau))/(2*sqrt(a*b))) )
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
Length  = idm_para.Length;  % vehicle length (m)

slag1   = Z(:,1);
u1      = u(1,:); % input from the leading vehicle
u2      = u(2,:); % input from the leading vehicle
s_d     = [ slag1(2) + slag1(3);
    a*(1 - (slag1(2)/v0)^sigma - ...
    (distance(slag1(2),u2,a,b,T,s0)/(u1))^2);...
    0];
end

function s_d = CF_idm2_bias_corrt(s,u,idm_para)
% This function inplements IDM CF model with time delay t and bias correction.
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
Length  = idm_para.Length;  % vehicle length (m)

u1      = u(1,:);           % input from the leading vehicle
u2      = u(2,:);           % input from the leading vehicle
s_d     = [ s(2)+s(3);
    a*(1 - (s(2)/v0)^sigma - ...
    (distance(s(2),u2,a,b,T,s0)/(u1))^2);...
    0];
end