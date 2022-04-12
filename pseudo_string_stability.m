clear; close all;
% IDM CF model parameter===================================================
idm_para.a = 1;      % maximum acceleration
idm_para.b = 2;      % comfortable deceleration
idm_para.sigma = 4;     % acceleration exponent
idm_para.s0 = 2;        % minimum distance (m)
idm_para.T = 1.1;       % safe time headway (s)
idm_para.v0 = 33.33;       % desired velocity (m/s)
idm_para.a_max = -0.2;   % max acceleration of random term
idm_para.a_min = -0.4;  % max deceleration of random term
idm_para.Length = 5;    % vehicle length (m)
idm_para.tau_var = 0;    % variance of random time delay
%==========================================================================
%   PlatoonConfig:
PlatoonConfig.N_platoon = 10;
% PlatoonConfig.alpha = [1];
PlatoonConfig.alpha = [0.7, 0.2, 0.1];
PlatoonConfig.beta = [0.7, 0.2, 0.1];
PlatoonConfig.s_e = 30;
PlatoonConfig.v_e = eq_h(idm_para, PlatoonConfig.s_e);
PlatoonConfig.perturbation = false;

A = -10; B = 25; C = -10;
tau1 = 0.0; tau2 = 3;

l2_1 = lambda2(A, B, C, tau1, tau2, idm_para, PlatoonConfig);

A = 0; B = 0; C = 0;
l2_2 = lambda2(A, B, C, tau1, tau2, idm_para, PlatoonConfig);

p_vector = [linspace(0, 1, 100)', 1 - linspace(0, 1, 100)'];
l2_avg = p_vector * [l2_2; l2_1];

x_intersec = interp1(l2_avg, linspace(0, 1, 100), 0);
plot(linspace(0, 1, 100), l2_avg, "LineWidth", 2, "Color", "blue")
grid on; hold on
xline(x_intersec, "LineWidth", 2, "Color", "r")
x_lim = xlim; y_lim = ylim;
fill([x_lim(1), x_lim(1), x_lim(2), x_lim(2)], [y_lim(2), 0, 0 ,y_lim(2)], [0.5 0.5 0.5]);
hold off
xlabel("Detection Sensitivity p")
ylabel("E_p[\lambda_2]")
alpha(.5)

%==========================================================================
function fv = pd_v(A, B, C, idm_para, PlatoonConfig)
a = idm_para.a; % maximum acceleration
b = idm_para.b; % comfortable deceleration
% sigma = idm_para.sigma; % acceleration exponent
s0 = idm_para.s0; % minimum distance (m)
T = idm_para.T; % safe time headway (s)
v0 = idm_para.v0; % desired velocity (m/s)
% Length = idm_para.Length; % vehicle length

v_e = PlatoonConfig.v_e;

fv = -4*a * (v_e + A)^3 / v0^4 - 2*a * ...
    (s0 + T*(v_e + A) + C*(v_e + A)/(2*sqrt(a*b)))...
    /(v_e + B)^2 * (T - C/(2*sqrt(a*b)));
end

function fs = pd_s(A, B, C, idm_para, PlatoonConfig)
a = idm_para.a; % maximum acceleration
b = idm_para.b; % comfortable deceleration
% sigma = idm_para.sigma; % acceleration exponent
s0 = idm_para.s0; % minimum distance (m)
T = idm_para.T; % safe time headway (s)
% v0 = idm_para.v0; % desired velocity (m/s)
% Length = idm_para.Length; % vehicle length

alpha1 = PlatoonConfig.alpha(1);
v_e = PlatoonConfig.v_e;
s_e = PlatoonConfig.s_e;
fs = alpha1 * 2*a * (s0 + T*(v_e + A) + C*(v_e + A)/(2*sqrt(a*b)))^2 ...
    /(s_e + B)^3;
end

function fdv = pd_dv(A, B, C, idm_para, PlatoonConfig)
a = idm_para.a; % maximum acceleration
b = idm_para.b; % comfortable deceleration
% sigma = idm_para.sigma; % acceleration exponent
s0 = idm_para.s0; % minimum distance (m)
T = idm_para.T; % safe time headway (s)
% v0 = idm_para.v0; % desired velocity (m/s)
% Length = idm_para.Length; % vehicle length

beta1 = PlatoonConfig.beta(1);
v_e = PlatoonConfig.v_e;
s_e = PlatoonConfig.s_e;
fdv = -beta1 * sqrt(a/b) * (v_e + A)/(s_e + B)^2 * ...
    (s0 + T*(v_e + A) + C*(v_e + A)/(2*sqrt(a*b)));
end


function ld1 = lambda1(A, B, C, idm_para, PlatoonConfig)
ld1 = pd_s(A, B, C, idm_para, PlatoonConfig) / ...
    pd_v(A, B, C, idm_para, PlatoonConfig);
end

function ld2 = lambda2(A, B, C, tau1, tau2, idm_para, PlatoonConfig)
alpha1 = PlatoonConfig.alpha(1);
alpha = PlatoonConfig.alpha;
beta1 = PlatoonConfig.beta(1);
% beta = PlatoonConfig.beta;

ld1 = lambda1(A, B, C, idm_para, PlatoonConfig);
fdv = pd_dv(A, B, C, idm_para, PlatoonConfig);
% fs = pd_s(A, B, C, idm_para, PlatoonConfig);
fv = pd_v(A, B, C, idm_para, PlatoonConfig);

ld2 = - ld1^2/fv - ld1^2 * tau1 + ld1^2 * fdv / fv * beta1 * (tau2 - tau1)...
    + ld1* fdv/fv + ld1 * (ld1 * (alpha1*tau1 + tau2 * sum(alpha(2:end)))...
     + [1:length(alpha)]*alpha' - 0.5);
end
