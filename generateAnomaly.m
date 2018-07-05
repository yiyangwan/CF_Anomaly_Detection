function [s_la, s_fa, AnomalyConfig] = generateAnomaly(s_l, s_f, AnomalyConfig)
% This function generate anomaly randomly based on the config parameters.
% Input:
%   s_l: baseline of the state squence of leading vehicle with dimension m
%       x n_sample
%   s_f: baseline of the state squence of following vehicle with dimension
%       m x n_sample
%   AnomalyConfig: 
%       .index: index of anomaly
%       .percent: percent of anomaly in scale [0,1]
%       .anomaly_type: list of anomaly types
%       .dur_length: the max duration of anomaly
%       .NoiseVar: Noise type anomaly covariance matrix with dimension m x m
%       .BiasVar: Bias type anomaly covariance matrix with dimension m x m
%       .DriftMax: Drift type anomaly max value

anomaly_type = AnomalyConfig.anomaly_type;
num_type = numel(anomaly_type); % number of anomaly types

s_la = s_l;
s_fa = s_f;

[m,n_sample] = size(s_l); % dimension of state, m, and number of samples, n_sample

AnomalyConfig.index = zeros(m,n_sample);
% Randomly generate the index of anomaly type
for i = 1:n_sample
    seed = rand(m,1);
    if(AnomalyConfig.index(1,i) ~= 1)&&(AnomalyConfig.index(2,i) ~= 1) % make sure anomaly will not overlapped
        if ((seed(1) <=AnomalyConfig.percent) || (seed(2) <=AnomalyConfig.percent))
            msk = (seed <= AnomalyConfig.percent);
            anomaly_type_idx = randi(num_type); % uniform distribution
            type = anomaly_type{anomaly_type_idx};
            dur_length = randi(AnomalyConfig.dur_length); % uniform distribution
        if (i+dur_length-1 <= n_sample)
            AnomalyConfig.index(:,i:i+dur_length-1) = AnomalyConfig.index(:,i:i+dur_length-1)+msk;
            
            switch type
                case 'Noise'
                    s_fa(:,i:i+dur_length-1) = s_fa(:,i:i+dur_length-1) + msk.*AnomalyConfig.NoiseVar * randn(m,dur_length);
                case 'Bias'
                    s_fa(:,i:i+dur_length-1) = s_fa(:,i:i+dur_length-1) + msk.*AnomalyConfig.BiasVar * randn(m,1);
                case 'Drift'
                    s_fa(:,i:i+dur_length-1) = s_fa(:,i:i+dur_length-1) + (2*randi(2,2,1)-3).*msk.*[linspace(0,randi(AnomalyConfig.DriftMax(1)),dur_length);linspace(0,randi(AnomalyConfig.DriftMax(2)),dur_length)];
            end
        end
        end
        
    end
        

    
end
AnomalyConfig.index = logical(AnomalyConfig.index);
end