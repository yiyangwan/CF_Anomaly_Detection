function [X, V, X_syn, V_syn] = platoon_model_v1(config, idm_para, N, s_0, alpha)
% config: config
% idm_para: IDM model parameter
% N: number of vehicles in platoon
% s_0: trajectory of platoon lead
% alpha: vector of weights, e.g. [0.7, 0.2, 0.1]

% get leading vehicle location x_l, speed v_l and acceleration a_l for training
[x_0, v_0] = data_process(s_0);

% number of samples
N_sample = length(x_0);
% number of cooperative vehicles
N_coop = length(alpha);

% offset of initial location of platoon lead
headway = 10;
gap = (N - 1) * headway;

if x_0(1) < gap
    x_0 = x_0 + gap - x_0(1);
end

% Generate following vehicle location x_f, speed v_f and acceleration a_l based on a
% car-following model

tau = config.tau; % human/sensor reaction time delay with unit "s"
delta_t = config.delta_t; % sampling time interval with unit "s"
t  = ceil(tau/delta_t); % time delay in discrete state-transition model

X = zeros(N, N_sample);
X(1, :) = x_0;
V = zeros(N, N_sample);
V(1, :) = v_0;

X_syn = zeros(N, N_sample);
V_syn = zeros(N, N_sample);
for i = 2 : N
    x_init = (N - i) * headway;
    v_init = 7.94;
    if i - 1 < N_coop
        % scale up weight vector
        alpha_t = alpha(1 : i - 1) / sum(alpha(1 : i - 1));
        x_l_syn = alpha_t * X(i - 1: -1: 1, :);
        v_l_syn = alpha_t * V(i - 1: -1: 1, :);
    else
        x_l_syn = alpha * X(i - 1 : -1 : i - N_coop, :);
        v_l_syn = alpha * V(i - 1 : -1 : i - N_coop, :);
    end
    s_f = cf_model(x_l_syn, v_l_syn, x_init, v_init, delta_t, t, tau, idm_para);
    X(i, :) = squeeze(s_f(:, 1));
    V(i, :) = squeeze(s_f(:, 2));
    X_syn(i, :) = x_l_syn;
    V_syn(i, :) = v_l_syn;
end