function [X, V, X_syn, V_syn] = platoon_model_v2(config, idm_para, ...
    PlatoonConfig, N_sample)
% Version 2: a ring configuration of platoon
%
% config: config
% idm_para: IDM model parameter
% N: number of vehicles in platoon
% N_sample: number of samples
% s_0: trajectory of platoon lead
% alpha: vector of weights, e.g. [0.7, 0.2, 0.1]
% headway: initial headway
% v_init: initial velocity

N_platoon = PlatoonConfig.N_platoon;
alpha = PlatoonConfig.alpha;
v_init = PlatoonConfig.v_init;
headway = PlatoonConfig.headway;
Length = idm_para.Length;
% number of cooperative vehicles
N_coop = length(alpha);

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

for j = t + 2 : N_sample
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

        X(i, j) = s(1); V(i, j) = s(2);

        X_syn(i, j) = x_l_syn;
        V_syn(i, j) = v_l_syn;
    end
end

