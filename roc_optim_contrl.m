% plot roc curve
%% ----------- tau = 0.5 --------------------------------------------------
close all
sen = [0,1,1,1,1,0.97758,0.97758,0.97758,0.97758,0.97309,0.96861,0.96861 ,0.96861,0.96413 ,0.9574 ];
fpR = [0,1,0.99718,0.91822,0.73773, 0.072194,0.07445,0.068246,0.067682,0.047941,0.0090243,0.0067682,0.0028201,0.001692 ,0.0003];

sen1 = [0,1,1,0.97758,0.96861,0.97309,0.96861,0.96861,0.95067,0.93722,0.91928];
fpR1 = [0,1,0.62831,0.12465,0.015228,0.041737,0.011844,0.0056402,0.0033841,0.001692,0.00056402];

sen2 = [0,1,0.99552,0.99103,0.98206,0.98206,0.97309,0.96861,0.93722,0.90135,0.713];
fpR2 = [0,1,0.8573,0.64354,0.51325,0.326,0.2634,0.17033,0.10998,0.084038,0.050197];

% plot(sort(fpR),sort(sen),'r','LineWidth',1); hold on;
% plot(sort(fpR1),sort(sen1),'b','LineWidth',1); 
% plot(sort(fpR2),sort(sen2),'m','LineWidth',1); 
% plot(0:1,0:1,'-.','LineWidth',1.5); xlim([0,0.25]);ylim([0.6,1])

xlabel('$1-specificity$', 'Interpreter','latex'), ylabel('$sensitivity$','Interpreter','latex')


% [1,1]
% ocsvm
sen3 = [0,1,1,1,1,0.97758,0.97758,0.96413,0.93274,0.9651];
fpR3= [0,1,0.99662,0.94078,0.77609,0.12183,0.23801,0.0067682,0.0022561,0.0050761];

% chi
sen4 = [0,1,1,0.97309 ,0.96861,0.97758,0.95964,0.73543,0.83857];
fpR4 = [0,1,0.63226,0.042301,0.011844,0.12465,0.0050761,0.0022561,0.0033841];

% chi without idm
sen5 = [0,1,0.99552,0.97758,0.99103,0.97309,0.93722,0.78924,0.63229,0.43946,0.26906];
fpR5 = [0,1,0.85787,0.47547,0.64411,0.2758,0.20192,0.13706,0.085166,0.059786,0.048505];

figure
plot(sort(fpR3),sort(sen3),'r:','LineWidth',1.6); hold on
plot(sort(fpR4),sort(sen4),'b:','LineWidth',1.6); 
plot(sort(fpR5),sort(sen5),'m:','LineWidth',1.6); hold off

auc.AUC1_OCSVM = trapz(sort(fpR3),sort(sen3));
auc.AUC1_chi_idm = trapz(sort(fpR4),sort(sen4));
auc.AUC1_chi = trapz(sort(fpR5),sort(sen5));

xlim([0,0.35]);ylim([0.1,1])
xlabel('$1-specificity$', 'Interpreter','latex'), ylabel('$sensitivity$','Interpreter','latex')

c1 = 'OCSVM with IDM ([1,1])';
c2 = '$\chi^2$-detector with IDM ([1,1])';
c3 = '$\chi^2$-detector without IDM ([1,1])';
legend(c1,c2,c3,'Location','southeast','Interpreter','latex');
% [0.1,0.1]
% ocsvm
sen6 = [0,1,0.97758,0.97758,0.93031,0.90135,0.91583,0.84305,0.83857,0.65022,0.37668];
fpR6 = [0,1,0.76311,0.63226,0.12521,0.070502,0.21207,0.032149,0.013536,0.00225610,.0028201];
% chi
sen7 = [0,1,0.97758,0.91031,0.88789,0.86547,0.69058,0.63229,0.55157,0.30942,0.1704];
fpR7 = [0,1,0.63226,0.12521,0.056966,0.025381,0.012972,0.01128,0.0073322,0.0050761,0.0028201];
% chi without idm
sen8 = [0,1,0.99103,0.97309,0.95516,0.93722,0.92825,0.9148,0.89238,0.8296,...
    0.72197,0.60987,0.46637,0.33632,0.19283,0.09417];
fpR8 = [0,1,0.92065,0.87001,0.77828,0.67079,0.61677,0.58301,0.50253,0.39111,...
    0.27968,0.22904,0.1632,0.10523,0.069218,0.037704];

figure
plot(sort(fpR6),sort(sen6),'r--','LineWidth',1.6); hold on
plot(sort(fpR7),sort(sen7),'b--','LineWidth',1.6); 
plot(sort(fpR8),sort(sen8),'m--','LineWidth',1.6); hold off

auc.AUC2_OCSVM = trapz(sort(fpR6),sort(sen6));
auc.AUC2_chi_idm = trapz(sort(fpR7),sort(sen7));
auc.AUC2_chi = trapz(sort(fpR8),sort(sen8));

c4 = 'OCSVM with IDM ([0.1,0.1])';
c5 = '$\chi^2$-detector with IDM ([0.1,0.1])';
c6 = '$\chi^2$-detector without IDM ([0.1,0.1])';
legend(c4,c5,c6,'Location','southeast','Interpreter','latex');
% [0.05,0.05]
% ocsvm
sen9 = [0,1,1,0.98655,0.97758,0.94531,0.93808,0.90341,0.87444,0.90135,0.80269,0.78027,0.43946];
fpR9 = [0,1,0.99436,0.93119,0.76537,0.39368,0.12521,0.066554,0.21094,0.39425,0.034405,0.014664,0.0045121];
% chi
sen10 = [0,1,0.97309,0.95067,0.92825,0.89238,0.85202,0.75785,0.67265,...
    0.6009,0.50224,0.23767];
fpR10 = [0,1,0.79629,0.63534,0.42037,0.22735,0.10636,0.059088,0.034328,...
    0.019696,0.010129,0.0050647];
% chi without idm
sen11 = [0,1,0.99103,0.98655,0.96413,0.92825,0.91928,0.84305,0.76682,0.6278,...
    0.5426,0.38565,0.2287,0.080717];
fpR11 = [0,1,0.92065,0.87001,0.77828,0.67192,0.58357,0.5031,0.39167,0.27912,...
    0.22847,0.16376,0.10523,0.070343];

figure
plot(sort(fpR9),sort(sen9),'r-','LineWidth',1.6); hold on
plot(sort(fpR10),sort(sen10),'b-','LineWidth',1.6); 
plot(sort(fpR11),sort(sen11),'m-','LineWidth',1.6); grid on,hold off

auc.AUC3_OCSVM = trapz(sort(fpR9),sort(sen9));
auc.AUC3_chi_idm = trapz(sort(fpR10),sort(sen10));
auc.AUC3_chi = trapz(sort(fpR11),sort(sen11));

c7 = 'OCSVM with IDM ([0.05,0.05])';
c8 = '$\chi^2$-detector with IDM ([0.05,0.05])';
c9 = '$\chi^2$-detector without IDM ([0.05,0.05])';
legend(c7,c8,c9,'Location','southeast','Interpreter','latex');

disp(auc);


%% ----------- tau = 0 ----------------------------------------------------
clear
xlabel('$1-specificity$', 'Interpreter','latex'), ylabel('$sensitivity$','Interpreter','latex')
% [1,1]
% ocsvm
sen3 = [0,1,0.91031,1,0.99552,0.99552,0.99103,0.97758,0.96413,0.96413,...
    0.95964,0.94619,0.9417];
fpR3= [0,1,0.00056275,0.97411,0.89364,0.91503,0.72369,0.35453,0.14744,0.06753,...
    0.032639,0.010129,0.0073157];

% chi
sen4 = [0,1,0.59476,0.99285,0.98808,0.98153,0.97259,0.96484,0.94934,0.92789,0.89154,0.84088,0.78129,0.71573,0.64243,...
    0.52741,0.40524];
fpR4 = [0,1,0.046107,0.95163,0.90325,0.84278,0.76115,0.67045,0.56085,0.4263,0.3031,0.21013,0.14135,0.089191,0.052154,...
    0.027211,0.016629];

% chi without idm
sen5 = [0,1,0.93623,0.92133,0.89869,0.87187,0.83313,0.78188,0.72169,0.65554,0.5733];
fpR5 = [0,1,0.43084,0.32048,0.22676,0.15268,0.095238,0.058957,0.035525,0.019652,0.0083144];

figure
plot(sort(fpR3),sort(sen3),'r:','LineWidth',1.6); hold on
plot(sort(fpR4),sort(sen4),'b:','LineWidth',1.6); 
plot(sort(fpR5),sort(sen5),'m:','LineWidth',1.6); hold off

auc.AUC1_OCSVM = trapz(sort(fpR3),sort(sen3));
auc.AUC1_chi_idm = trapz(sort(fpR4),sort(sen4));
auc.AUC1_chi = trapz(sort(fpR5),sort(sen5));

xlim([0,0.35]);ylim([0.1,1])
xlabel('$1-specificity$', 'Interpreter','latex'), ylabel('$sensitivity$','Interpreter','latex')

c1 = 'OCSVM with IDM ([1,1])';
c2 = '$\chi^2$-detector with IDM ([1,1])';
c3 = '$\chi^2$-detector without IDM ([1,1])';
legend(c1,c2,c3,'Location','southeast','Interpreter','latex');
% [0.1,0.1]
% ocsvm
sen6 = [0,1,1,0.99552,0.99552,0.99103,0.98206,0.95067,0.93722,...
    0.90135,0.89686,0.88789,0.84305,0.78924];
fpR6 = [0,1,0.99662,0.91503,0.89026,0.83737,0.77997,0.5408,0.34947,...
    0.087788,0.087226,0.046708,0.010129,0.0011255 ];
% chi
sen7 = [0,1,0.95964,0.98655,0.93274,0.91031,0.88341,0.88789,0.82511,0.79821,...
    0.78924];
fpR7 = [0,1,0.60945,0.78897,0.37985,0.12662,0.042206,0.061339,0.013506,...
    0.0033765,0.0016882];
% chi without idm
sen8 = [0,1,0.98655,0.95964,0.93722,0.90583,0.88341,0.86099,0.85202,...
    0.83408,0.65919,0.61883,0.60538,0.44843,0.29596];
fpR8 = [0,1,0.86719,0.64266,0.55262,0.39899,0.30388,0.26337,0.24086,0.17501,...
    0.10411,0.077659,0.059651,0.030388,0.016882];

figure
plot(sort(fpR6),sort(sen6),'r--','LineWidth',1.6); hold on
plot(sort(fpR7),sort(sen7),'b--','LineWidth',1.6); 
plot(sort(fpR8),sort(sen8),'m--','LineWidth',1.6); hold off

auc.AUC2_OCSVM = trapz(sort(fpR6),sort(sen6));
auc.AUC2_chi_idm = trapz(sort(fpR7),sort(sen7));
auc.AUC2_chi = trapz(sort(fpR8),sort(sen8));

c4 = 'OCSVM with IDM ([0.1,0.1])';
c5 = '$\chi^2$-detector with IDM ([0.1,0.1])';
c6 = '$\chi^2$-detector without IDM ([0.1,0.1])';
legend(c4,c5,c6,'Location','southeast','Interpreter','latex');

% [0.05,0.05]
% ocsvm
sen9 = [0,1,1,0.99552,0.99103,0.99103,0.98206,0.95964,0.90135,...
    0.84753,0.80269,0.78924,0.78027,0.85202,0.75785];
fpR9 = [0,1,0.99606,0.98537,0.9668,0.87957,0.80416,0.66742,0.34834,...
    0.13225,0.061339,0.018008,0.011818,0.1345,0.010692];
% chi
sen10 = [0,1,0.56502,0.95516,0.97309,0.91031,0.88789,0.82511,0.7713...
    ,0.66368];
fpR10 = [0,1,0.002251,0.61227,0.7901,0.38436,0.1677,0.062465,0.020259...
    ,0.0073157];
% chi without idm
sen11 = [0,1,0.99103,0.98206,0.97309,0.92825,0.91928,0.83857,0.76233,...
    0.66368,0.50224,0.39013,0.24215,0.09417,0.040359];
fpR11 = [0,1,0.92178,0.86775,0.77603,0.66742,0.58188,0.49859,0.39055,...
    0.27406,0.22735,0.16095,0.10467,0.069218,0.036579];

figure
plot(sort(fpR9),sort(sen9),'r-','LineWidth',1.6); hold on
plot(sort(fpR10),sort(sen10),'b-','LineWidth',1.6); 
plot(sort(fpR11),sort(sen11),'m-','LineWidth',1.6); grid on,hold off

auc.AUC3_OCSVM = trapz(sort(fpR9),sort(sen9));
auc.AUC3_chi_idm = trapz(sort(fpR10),sort(sen10));
auc.AUC3_chi = trapz(sort(fpR11),sort(sen11));

c7 = 'OCSVM with IDM ([0.05,0.05])';
c8 = '$\chi^2$-detector with IDM ([0.05,0.05])';
c9 = '$\chi^2$-detector without IDM ([0.05,0.05])';
legend(c7,c8,c9,'Location','southeast','Interpreter','latex');

disp(auc);

%% ----------- tau = 1 ----------------------------------------------------
clear
xlabel('$1-specificity$', 'Interpreter','latex'), ylabel('$sensitivity$','Interpreter','latex')
% [1,1]
% ocsvm
sen3 = [0,1,1,1,0.99552,0.99552,0.99103,0.98206,0.97309,...
    0.95516,0.9148,0.83857,0.8565,0.95516];
fpR3 = [0,1,0.99381,0.98424,0.96624,0.92515,0.81823,0.60664,0.33596,...
    0.15194,0.0287,0.0073157,0.013506,0.19302];

% chi
sen4 = [0,1,0.98655,0.98655,0.97758,0.97309,0.96861,0.96861,0.95067,...
    0.93722,0.92377,0.92377,0.84305,0.79821,0.67713];
fpR4 = [0,1,0.8188,0.68092,0.49522,0.31626,0.19471,0.12718,0.088351,...
    0.061339,0.039392,0.039392,0.021384,0.012943,0.004502];

% chi without idm
sen5 = [0,1,1,0.98206,0.96861,0.96413,0.96413,0.94619,0.92825,0.90583,...
    0.90135,0.86996,0.72646,0.65471,0.55157];
fpR5 = [0,1,0.92515,0.87001,0.78165,0.67811,0.58751,0.50478,0.40293,0.28475,...
    0.22904,0.16207,0.10861,0.069218,0.040518];

figure
plot(sort(fpR3),sort(sen3),'r:','LineWidth',1.6); hold on
plot(sort(fpR4),sort(sen4),'b:','LineWidth',1.6); 
plot(sort(fpR5),sort(sen5),'m:','LineWidth',1.6); hold off

auc.AUC1_OCSVM = trapz(sort(fpR3),sort(sen3));
auc.AUC1_chi_idm = trapz(sort(fpR4),sort(sen4));
auc.AUC1_chi = trapz(sort(fpR5),sort(sen5));

xlim([0,0.35]);ylim([0.1,1])
xlabel('$1-specificity$', 'Interpreter','latex'), ylabel('$sensitivity$','Interpreter','latex')

c1 = 'OCSVM with IDM ([1,1])';
c2 = '$\chi^2$-detector with IDM ([0.1,0.1])';
c3 = '$\chi^2$-detector without IDM ([0.1,0.1])';
legend(c1,c2,c3,'Location','southeast','Interpreter','latex');
% [0.1,0.1]
% ocsvm
sen6 = [0,1,1,0.99103,0.98206,0.96861,0.95067,0.9148,0.91031,0.89686,...
    0.86099,0.85202,0.47982,0.89686,0.81614,0.82511];
fpR6 = [0,1,0.991,0.97243,0.92572,0.82048,0.68486,0.57625,0.3686,0.33315,...
    0.20259,0.14575,0.028137,0.0078784,0.084975,0.11311];
% chi
sen7 = [0,1,0.97758,0.97309,0.95964,0.93722,0.91928,0.89686,0.80717,0.77578,...
    0.68161,0.58744,0.39013,0.21076];
fpR7 = [0,1,0.81992,0.68374,0.49747,0.31795,0.1964,0.1525,0.10355,0.073157,...
    0.046708,0.028137,0.015757,0.006753];
% chi without idm
sen8 = [0,1,0.99103,0.98206,0.96861,0.95516,0.91928,0.90583,0.81614,...
    0.73094,0.64574,0.4843,0.32287,0.19731,0.13004];
fpR8 = [0,1,0.9184,0.86832,0.77603,0.68036,0.58976,0.50816,0.40518,...
    0.28756,0.2296,0.16545,0.10917,0.070906,0.040518];

figure
plot(sort(fpR6),sort(sen6),'r--','LineWidth',1.6); hold on
plot(sort(fpR7),sort(sen7),'b--','LineWidth',1.6); 
plot(sort(fpR8),sort(sen8),'m--','LineWidth',1.6); hold off

auc.AUC2_OCSVM = trapz(sort(fpR6),sort(sen6));
auc.AUC2_chi_idm = trapz(sort(fpR7),sort(sen7));
auc.AUC2_chi = trapz(sort(fpR8),sort(sen8));

c4 = 'OCSVM with IDM ([0.1,0.1])';
c5 = '$\chi^2$-detector with IDM ([0.1,0.1])';
c6 = '$\chi^2$-detector without IDM ([0.1,0.1])';
legend(c4,c5,c6,'Location','southeast','Interpreter','latex');

% [0.05,0.05]
% ocsvm
sen9 = [0,1,1,1,0.99552,0.97758,0.92377,0.9417,0.9417,...
    0.89238,0.8296,0.81614,0.80717,0.79821,0.7713,0.69507,0.53363,...
    0.33184];
fpR9 = [0,1,0.99887,0.99775,0.98199,0.861,0.41868,0.66629,0.58582,...
    0.27068,0.19921,0.1525,0.1238,0.10242,0.087788,0.059651,0.017445,...
    0.0084412];
% chi
sen10 = [0,1,0.96861,0.95964,0.93722,0.91031,0.86996,0.78475,0.73543,...
    0.66368,0.53363,0.27803,0.19283];
fpR10 = [0,1,0.81992,0.68374,0.49803,0.31851,0.19752,0.12943,0.090039,...
    0.062465,0.042206,0.021947,0.012943];
% chi without idm
sen11 = [0,1,0.98655,0.97309,0.95964,0.9417,0.92377,0.85202,0.77578,...
    0.63229,0.50224,0.38117,0.22422,0.11211,0.040359];
fpR11 = [0,1,0.92797,0.87282,0.78278,0.68092,0.58976,0.50816,0.40574,...
    0.28813,0.23241,0.16545,0.10974,0.070906,0.039955];

figure
plot(sort(fpR9),sort(sen9),'r-','LineWidth',1.6); hold on
plot(sort(fpR10),sort(sen10),'b-','LineWidth',1.6); 
plot(sort(fpR11),sort(sen11),'m-','LineWidth',1.6); grid on,hold off

auc.AUC3_OCSVM = trapz(sort(fpR9),sort(sen9));
auc.AUC3_chi_idm = trapz(sort(fpR10),sort(sen10));
auc.AUC3_chi = trapz(sort(fpR11),sort(sen11));

c7 = 'OCSVM with IDM ([1,1])';
c8 = '$\chi^2$-detector with IDM ([0.05,0.05])';
c9 = '$\chi^2$-detector without IDM ([0.05,0.05])';
legend(c7,c8,c9,'Location','southeast','Interpreter','latex');

disp(auc);
