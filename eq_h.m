function sol = eq_h(idm_para, h)
s0      = idm_para.s0;      % minimum distance (m)
T       = idm_para.T;       % safe time headway (s)
v0      = idm_para.v0;      % desired velocity (m/s)


eq_headway = @(eq_v) (s0 + T * eq_v) * (1 - (eq_v / v0)^4)^(-0.5) - h;
sol = fzero(eq_headway, 1);
fprintf('Equilibrium velocity: %.3f m/s\n', sol)
fprintf('Equilibrium headway: %.3f m\n', h)