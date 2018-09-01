% plot roc curve
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

%% [1,1]
%ocsvm
sen3 = [0,1,1,1,1,0.97758,0.97758,0.96413,0.93274,0.9651];
fpR3= [0,1,0.99662,0.94078,0.77609,0.12183,0.23801,0.0067682,0.0022561,0.0050761];

% chi
sen4 = [0,1,1,0.97309 ,0.96861,0.97758,0.95964,0.73543,0.83857   ];
fpR4 = [0,1,0.63226,0.042301,0.011844,0.12465,0.0050761,0.0022561,0.0033841];

% chi without idm
sen5 = [0,1,0.99552,0.97758,0.99103,0.97309,0.93722,0.78924,0.63229,0.43946,0.26906];
fpR5 = [0,1,0.85787,0.47547,0.64411,0.2758,0.20192,0.13706,0.085166,0.059786,0.048505];


plot(sort(fpR),sort(sen),'r:','LineWidth',1.6); hold on
plot(sort(fpR1),sort(sen1),'b:','LineWidth',1.6); 
plot(sort(fpR5),sort(sen5),'m:','LineWidth',1.6); 

xlim([0,0.35]);ylim([0.1,1])
xlabel('$1-specificity$', 'Interpreter','latex'), ylabel('$sensitivity$','Interpreter','latex')
%% [0.1,0.1]
%ocsvm
sen6 = [0,1,0.97758,0.97758,0.93031,0.90135,0.91583,0.84305,0.83857,0.65022,0.37668];
fpR6 = [0,1,0.76311,0.63226,0.12521,0.070502,0.21207,0.032149,0.013536,0.00225610,.0028201];
% chi
sen7 = [0,1,0.97758,0.91031,0.88789,0.86547,0.69058,0.63229,0.55157,0.30942,0.1704];
fpR7 = [0,1,0.63226,0.12521,0.056966,0.025381,0.012972,0.01128,0.0073322,0.0050761,0.0028201];
% chi without idm
sen8 = [0,1,0.23767,0.95516,0.92377,0.86996,0.70404,0.3991,0.15695];
fpR8 = [0,1,0.10434,0.85787,0.64523,0.47547,0.27806,0.14044,0.089114];

plot(sort(fpR6),sort(sen6),'r--','LineWidth',1.6); 
plot(sort(fpR7),sort(sen7),'b--','LineWidth',1.6); 
plot(sort(fpR8),sort(sen8),'m--','LineWidth',1.6); 

%% [0.05,0.05]
%ocsvm
sen9 = [0,1,1,0.98655,0.97758,0.94531,0.93808,0.90341,0.87444,0.90135,0.80269,0.78027,0.43946];
fpR9 = [0,1,0.99436,0.93119,0.76537,0.39368,0.12521,0.066554,0.21094,0.39425,0.034405,0.014664,0.0045121];
% chi
sen10 = [0,1,0.95516,0.92825,0.88341,0.8296,0.71749,0.61435,0.56951,0.50224,0.21525];
fpR10 = [0,1,0.63226,0.40666,0.12521,0.043429,0.019741,0.012972,0.01128,0.0062042,0.0056402];
% chi without idm
sen11 = [0,1,0.96861,0.91928,0.89238,0.82063,0.70404,0.56502,0.49776,0.3991,0.27803,0.21973,0.09417];
fpR11 = [0,1,0.85787,0.64523,0.55612,0.47603,0.32769,0.27975,0.23463,0.17484,0.13988,0.11224,0.089114];

plot(sort(fpR9),sort(sen9),'r-','LineWidth',1.6); 
plot(sort(fpR10),sort(sen10),'b-','LineWidth',1.6); 
plot(sort(fpR11),sort(sen11),'m-','LineWidth',1.6); grid on

c1 = 'OCSVM with IDM ([1,1])';
c2 = '$\chi^2$-detector with IDM ([1,1])';
c3 = '$\chi^2$-detector without IDM ([1,1])';

c4 = 'OCSVM with IDM ([0.1,0.1])';
c5 = '$\chi^2$-detector with IDM ([0.1,0.1])';
c6 = '$\chi^2$-detector without IDM ([0.1,0.1])';


c7 = 'OCSVM with IDM ([0.05,0.05])';
c8 = '$\chi^2$-detector with IDM ([0.05,0.05])';
c9 = '$\chi^2$-detector without IDM ([0.05,0.05])';
legend(c1,c2,c3,c4,c5,c6,c7,c8,c9,'Location','southeast','Interpreter','latex');