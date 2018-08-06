clear
clc

dataPath = 'C:\Users\SQwan\Documents\MATLAB\CF\dataset\'; % dataset location
load(strcat(dataPath,'testdata.mat')) % info of the leading vehicle = s for testing n_sample * m
load(strcat(dataPath,'rawdata.mat')) % info of the leading vechicle = s_train for training n_sample * m
raw_data   = s_train;

ini = IniConfig();
ini.ReadFile('config.ini');
sections = ini.GetSections();

[ms1,ns1] = size(sections); % ms1 be the length of config type, ns1 should be 1
keys = cell([ns1,ms1]);
count_keys = cell([ns1,ms1]);
count_keys1 = cell([ns1,ms1]);
values = cell([ns1,ms1]);
Summary = cell([ns1,ms1]);

for i = 1 : ms1
    [keys{i}, count_keys{i}] = ini.GetKeys(sections{i});

    values{i} = ini.GetValues(sections{i},keys{i});
    count_keys1{i} = 1:count_keys{i};

end

AllCombine = allcomb(count_keys1{:}); % enumerate all combinations of parameters

[m1,~] = size(AllCombine);

% Config data structure====================================================
config.R = diag([1,0.5]);
config.Q = diag([0.8,0.05]);
config.H = eye(2);
config.delta_t = 0.1;
config.adptQ = true;
config.adptR = false;
config.use_predict = false;

weight_vector = [3,7];
config.weight = weight_vector./sum(weight_vector); % weights for AdEKF when time window size is 2

% IDM CF model parameter===================================================
idm_para.a = 0.73; % maximum acceleration
idm_para.b = 1.67; % comfortable deceleration
idm_para.sigma = 4; % acceleration exponent 
idm_para.s0 = 2; % minimum distance (m)
idm_para.T = 1.5; % safe time headway (s)
idm_para.v0 = 24; % desired velocity (m/s)
idm_para.a_max = 0.0; % max acceleration of random term 
idm_para.a_min = -0.0; % max deceleration of random term

AnomalyConfig.anomaly_type = {'Noise','Bias','Drift'};
AnomalyConfig.seed = 10; % random seed controler

for i = 1:m1
    % Config data structure================================================
    config.OCSVM = values{1}{AllCombine(i,1)};
    config.OCSVM_threshold = values{2}{AllCombine(i,2)};

    config.N = values{3}{AllCombine(i,3)};
    config.r = values{4}{AllCombine(i,4)};
    
    config.tau = values{5}{AllCombine(i,5)};
    config.N_ocsvm = values{6}{AllCombine(i,6)};
    
    % Anomaly parameters
    
    AnomalyConfig.percent = values{7}{AllCombine(i,7)};
    AnomalyConfig.dur_length = values{8}{AllCombine(i,8)};
    
    AnomalyConfig.NoiseVar = diag(values{9}{AllCombine(i,9)});
    AnomalyConfig.BiasVar = diag(values{10}{AllCombine(i,10)});
    AnomalyConfig.DriftMax = values{11}{AllCombine(i,11)};
    
    config.use_CF = values{12}{AllCombine(i,12)}; % use CF model or not
    
    Summary{i} = mainFunction(config,idm_para,AnomalyConfig,raw_data,s);
    path = 'D:\research\CF_detection\results\'; % result files location
    filename = strcat( path , sprintf('Summary_%d.txt',i)); 
    writetable(Summary{i}.results,filename)
    
    fprintf('Run %d:\n',i);
    disp(config); disp(AnomalyConfig);
    fprintf('results');
    disp(Summary{i}.results);
end  
