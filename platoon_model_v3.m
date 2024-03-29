function [X, V, X_syn, V_syn, AnomalyConfig] = platoon_model_v3(config, idm_para, ...
    PlatoonConfig, N_sample, AnomalyConfig)
% Version 3: Add anomaly injection function
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
% AnomalyConfig: anomaly config (optional)

N = PlatoonConfig.N_platoon;
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

X = zeros(N, N_sample);
X(:, 1) = (headway + Length) * [N - 1: -1 : 0]';
V = zeros(N, N_sample) + v_init;

if t > 1
    for tt = 2 : t + 1
        X(:, tt) = X(:, tt - 1) + V(:, tt - 1) * config.delta_t;
    end
end

% Add perturbation
if PlatoonConfig.perturbation
    X(1, 1 + t) = X(1, 1 + t) - 0.5 * headway;
%     V(1, 1 + t) = V(1, 1 + t) + 0.5 * V(1, 1 + t);
end

X_syn = zeros(N, N_sample);
V_syn = zeros(N, N_sample);

if PlatoonConfig.inject_anomaly && exist("AnomalyConfig", "var")
    rng(AnomalyConfig.seed); % control random number generator so that it can produce predictable random numbers
    anomaly_type = AnomalyConfig.anomaly_type;
    num_type = numel(anomaly_type); % number of anomaly types
    anomaly_sequence = zeros(2, N_sample);
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


for j = t + 2 : N_sample
    for i = 1 : N
        tau1 = normrnd(tau,idm_para.tau_var);

        if i - 1 < N_coop
            X_temp = [X(i - 1 : -1 : 1, j - t - 1); X(N: -1 : N - N_coop + i, j - t - 1) + (headway + Length) + X(1, 1)];
            V_temp = [V(i - 1 : -1 : 1, j - t - 1); V(N: -1 : N - N_coop + i, j - t - 1)];
            x_l_syn = alpha * X_temp;
            v_l_syn = alpha * V_temp;
        else
            x_l_syn = alpha * X(i - 1 : -1 : i - N_coop, j - t - 1);
            v_l_syn = alpha * V(i - 1 : -1 : i - N_coop, j - t - 1);
        end

        s = idm(x_l_syn,v_l_syn,X(i, j - 1),V(i, j - 1),delta_t,t,tau1,idm_para);

        if exist("anomaly_sequence", "var") && i == PlatoonConfig.attack_id
            s = s + anomaly_sequence(:, j);
        end
        X(i, j) = s(1); V(i, j) = s(2);

        X_syn(i, j) = x_l_syn;
        V_syn(i, j) = v_l_syn;
    end
end

