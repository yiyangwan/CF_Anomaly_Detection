clear
clc

dataPath = 'D:\Documents\MATLAB\CF_Anomaly_Detection\dataset\'; % dataset location
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

AllCombine = allcomb(count_keys1{1:13}); % enumerate all combinations of parameters

[m1,~] = size(AllCombine);

% Config data structure====================================================
config.delta_t = 0.1;
config.adptQ = false;
config.adptR = false;
config.R     = diag([0.01,0.01]);    % observation noise covariance
config.use_predict = false;
config.ukf = false;
config.print = false;
config.detection = true;

weight_vector = [3,7];
config.weight = weight_vector./sum(weight_vector); % weights for AdEKF when time window size is 2

% IDM CF model parameter===================================================
idm_para.a = 0.73; % maximum acceleration
idm_para.b = 1.67; % comfortable deceleration
idm_para.sigma = 4; % acceleration exponent 
idm_para.s0 = 2; % minimum distance (m)
idm_para.T = 1.5; % safe time headway (s)
idm_para.v0 = 24; % desired velocity (m/s)
idm_para.a_max = -0.4; % max acceleration of random term 
idm_para.a_min = -0.9; % max deceleration of random term
idm_para.Length = 0; % vehicle length (m)
idm_para.tau_var =1;    % variance of random time delay

AnomalyConfig.anomaly_type = {'Noise','Bias','Drift'};

seed = [1:1:20]; % random seed controler
bias_correct = [values{1,14}{:}];

for jj = 1:2
    config.bias_correct = bias_correct(jj);
for ii = 1: max(size(seed))
    AnomalyConfig.seed = seed(ii);
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
    config.use_predict = values{13}{AllCombine(i,13)}; % replace estimate as predict when anomaly detected or not
    
%     config.bias_correct = values{14}{AllCombine(i,14)}; % whether use augmented EKF or not
    
    if(config.bias_correct)
        config.Q                = diag([0.5,0.3,1e1]);  %diag([0.5,0.3]);% process noise covariance
        config.H                = [1,0,1;0,1,0];    % observation matrix
    else
        config.Q                = diag([0.5,0.3]);  % process noise covariance
        config.H                = [1,0;0,1];        % observation matrix
    end
    
    Summary{i} = mainFunction(config,idm_para,AnomalyConfig,raw_data,s);
    path = 'D:\Documents\MATLAB\CF_Anomaly_Detection\results\'; % result files location
    filename = strcat( path , sprintf('Summary_%d_12_1_19.txt',i)); 
    writetable(Summary{i}.results,filename)
    
    fprintf('Run %d:\n',i);
    disp(config); disp(AnomalyConfig);
    fprintf('results');
    disp(Summary{i}.results);
    sen1(i) = table2array(Summary{i}.results(8,2));
    fpr1(i) = table2array(Summary{i}.results(10,2));
end  


%% Caculating AUC and ROC


sen1 = [sen1,0,1];
fpr1 = [fpr1,0,1];

% plot

% plot(sort(fpr1),sort(sen1),'LineWidth',1.8); 
% hold on
% xlim([0,0.4]);ylim([0.1,1])
% xlabel('$1-specificity$', 'Interpreter','latex'), ylabel('$sensitivity$','Interpreter','latex')   

auc(jj,ii) = trapz(sort(fpr1),sort(sen1));
end
end
avr = mean(auc,2)
stdd = std(auc,0,2)
[t1,h1] = ttest(auc(1,:),auc(2,:))
[t2,h2] = ttest2(auc(1,:),auc(2,:))