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
%ocsvm
sen3 = [0,1,1,1,1,0.97758,0.97758,0.96413,0.93274,0.9651];
fpR3= [0,1,0.99662,0.94078,0.77609,0.12183,0.23801,0.0067682,0.0022561,0.0050761];

% chi
sen4 = [0,1,1,0.97309 ,0.96861,0.97758,0.95964,0.73543,0.83857   ];
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
%ocsvm
sen6 = [0,1,0.97758,0.97758,0.93031,0.90135,0.91583,0.84305,0.83857,0.65022,0.37668];
fpR6 = [0,1,0.76311,0.63226,0.12521,0.070502,0.21207,0.032149,0.013536,0.00225610,.0028201];
% chi
sen7 = [0,1,0.97758,0.91031,0.88789,0.86547,0.69058,0.63229,0.55157,0.30942,0.1704];
fpR7 = [0,1,0.63226,0.12521,0.056966,0.025381,0.012972,0.01128,0.0073322,0.0050761,0.0028201];
% chi without idm
sen8 = [0,1,0.23767,0.95516,0.92377,0.86996,0.70404,0.3991,0.15695];
fpR8 = [0,1,0.10434,0.85787,0.64523,0.47547,0.27806,0.14044,0.089114];

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
%ocsvm
sen9 = [0,1,1,0.98655,0.97758,0.94531,0.93808,0.90341,0.87444,0.90135,0.80269,0.78027,0.43946];
fpR9 = [0,1,0.99436,0.93119,0.76537,0.39368,0.12521,0.066554,0.21094,0.39425,0.034405,0.014664,0.0045121];
% chi
sen10 = [0,1,0.95516,0.92825,0.88341,0.8296,0.71749,0.61435,0.56951,0.50224,0.21525];
fpR10 = [0,1,0.63226,0.40666,0.12521,0.043429,0.019741,0.012972,0.01128,0.0062042,0.0056402];
% chi without idm
sen11 = [0,1,0.96861,0.91928,0.89238,0.82063,0.70404,0.56502,0.49776,0.3991,0.27803,0.21973,0.09417];
fpR11 = [0,1,0.85787,0.64523,0.55612,0.47603,0.32769,0.27975,0.23463,0.17484,0.13988,0.11224,0.089114];

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
%ocsvm
sen3 = [0,1,0.91031,1,0.99552,0.99552,0.99103,0.97758,0.96413,0.96413,...
    0.95964,0.94619,0.9417];
fpR3= [0,1,0.00056275,0.97411,0.89364,0.91503,0.72369,0.35453,0.14744,0.06753,...
    0.032639,0.010129,0.0073157];

% chi
sen4 = [0,1,0.89686,0.97758,0.96413,0.94619,0.92825,0.91928,0.97309];
fpR4 = [0,1,0.0033765,0.63647,0.1812,0.09229,0.011818,0.0050647,0.30107];

% chi without idm
sen5 = [0,1,0.91031,1,0.97309,0.95964,0.9417,0.92377,0.92377,0.9148,0.75785...
    ,0.70404,0.6861,0.61435,0.46188,0.30045];
fpR5 = [0,1,0.17389,0.98706,0.64209,0.55205,0.39899,0.30332,0.25661,0.24029,0.1193...
    ,0.081598,0.066404,0.052335,0.030388,0.017445];

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
%ocsvm
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
%ocsvm
sen9 = [0,1,1,0.98655,0.97758,0.94531,0.93808,0.90341,0.87444,0.90135,0.80269,0.78027,0.43946];
fpR9 = [0,1,0.99436,0.93119,0.76537,0.39368,0.12521,0.066554,0.21094,0.39425,0.034405,0.014664,0.0045121];
% chi
sen10 = [0,1,0.95516,0.92825,0.88341,0.8296,0.71749,0.61435,0.56951,0.50224,0.21525];
fpR10 = [0,1,0.63226,0.40666,0.12521,0.043429,0.019741,0.012972,0.01128,0.0062042,0.0056402];
% chi without idm
sen11 = [0,1,0.96861,0.91928,0.89238,0.82063,0.70404,0.56502,0.49776,0.3991,0.27803,0.21973,0.09417];
fpR11 = [0,1,0.85787,0.64523,0.55612,0.47603,0.32769,0.27975,0.23463,0.17484,0.13988,0.11224,0.089114];

figure
plot(sort(fpR9),sort(sen9),'r-','LineWidth',1.6); hold on
plot(sort(fpR10),sort(sen10),'b-','LineWidth',1.6); 
plot(sort(fpR11),sort(sen11),'m-','LineWidth',1.6); grid on,hold off

auc.AUC3_OCSVM = trapz(sort(fpR9),sort(sen9));
auc.AUC3_chi_idm = trapz(sort(fpR10),sort(sen10));
auc.AUC3_chi = trapz(sort(fpR11),sort(sen11));

c7 = 'OCSVM with IDM ([1,1])';
c8 = '$\chi^2$-detector with IDM ([1,1])';
c9 = '$\chi^2$-detector without IDM ([1,1])';
legend(c7,c8,c9,'Location','southeast','Interpreter','latex');

disp(auc);