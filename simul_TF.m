%
clear all
clc

% Define the parameter

alpha = [0.7, 0.2, 0.1];
beta = [0.7, 0.2, 0.1];
tau_1 = 0;
tau_2 = 0.5;
v_e = 22.47;
s_e = 30;
s_0 = 2;
v_0 = 33.33;
T = 1.1;
a = 1;
b = 2;

% Determine the noise terms with respect of A,B,C

A = -5;
B = 15;
C = -6;


% Given different detection rate p, check the largest eigenvalue.
% Solve for eigenvalues and their norms

N_1 = 101;
N_2 = 10000;
result = zeros(1, N_1);

[f_vn, f_dvn, f_sn] = derivative(0,0,0,alpha,beta,v_e,s_e,v_0,s_0,T,a,b);
[f_vn_tilde, f_dvn_tilde, f_sn_tilde] = derivative(A,B,C,alpha,beta,v_e,s_e,v_0,s_0,T,a,b);

i = 1;

for p  = linspace(0,1,N_1)

    result_eig = zeros(4,N_2);
    j = 1;

    for omega = linspace(0, pi, N_2)
        TF_matrix = p*TF(1i*omega, f_vn, f_dvn, f_sn, alpha, beta, tau_1, tau_2) + (1-p)*TF(1i*omega, f_vn_tilde, f_dvn_tilde, f_sn_tilde, alpha, beta, tau_1, tau_2);
        result_eig(:,j) = eig(TF_matrix);
        j = j+1;
    end

    norm_val_tilde = sort(abs(result_eig),'descend');
    [max_norm_val,index_tilde] = max(norm_val_tilde);
    result(1,i) = max(max_norm_val);
    i = i+1;
end


%%
close all
plot(linspace(0,1,N_1), result, LineWidth=2);
xlabel('$\tilde{p}$','FontWeight','bold', 'interpreter','latex');
% xlabel(h,'$p$','FontWeight','bold', 'interpreter','latex');
ylabel('$\sup |\lambda_k|$','FontWeight','bold', 'interpreter','latex');
title('The largest magnititude of eigenvalues', 'fontsize', 10);
grid on
grid minor
ylim([0,max(result)+1])








function [f_vn, f_dvn, f_sn] = derivative(A,B,C,alpha,beta,v_e,s_e,v_0,s_0,T,a,b)
    f_vn = -4*a*(v_e+A)^3/v_0^4 - 2*a*(T+C/sqrt(a*b))*(s_0+T*(v_e+A)+C*(v_e+A)/(2*sqrt(a*b)))/(s_e+B)^2;
    f_dvn = -beta(1)*sqrt(a/b)*(v_e+A)*(s_0+T*(v_e+A)+C*(v_e+A)/2*sqrt(a*b))/(s_e+B)^2;
    f_sn = 2*alpha(1)*a*(s_0+T*(v_e+A)+C*(v_e+A)/2*sqrt(a*b))^2/(s_e+B)^3;
end

function P_hat = TF(s, f_vn, f_dvn, f_sn, alpha, beta, tau_1, tau_2)

    T_1 = (s*(beta(2)-beta(1))*f_dvn+(alpha(1)-alpha(2))*f_sn)*exp(-s*tau_2)/(s^2-s*(f_vn+beta(1)*f_dvn)*exp(-s*tau_1)+alpha(1)*f_sn*exp(-s*tau_1));
    T_2 = (s*(beta(3)-beta(2))*f_dvn+(alpha(2)-alpha(3))*f_sn)*exp(-s*tau_2)/(s^2-s*(f_vn+beta(1)*f_dvn)*exp(-s*tau_1)+alpha(1)*f_sn*exp(-s*tau_1));
    T_3 = (-s*beta(3)*f_dvn+alpha(3)*f_sn)*exp(-s*tau_2)/(s^2-s*(f_vn+beta(1)*f_dvn)*exp(-s*tau_1)+alpha(1)*f_sn*exp(-s*tau_1));
    
    P_hat = [T_1 T_2 T_3 0;
             1   0   0   0
             0   1   0   0
             0   0   1   0];
end