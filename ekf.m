function [x_next,P_next,x_dgr,P_dgr,p,K,error1] = ekf(f,h,y,del_f,del_h,x_hat,P_hat,u,groundtruth,CF,dde_his,tk1,tk2,config,psum)
% Extended Kalman filter
%
% -------------------------------------------------------------------------
%
% State space model is
% X_k+1 = f_k(X_k,U_k) + V_k+1   -->  state update
% Y_k = h_k(X_k) + W_k       -->  measurement
% 
% V_k+1 zero mean uncorrelated gaussian, cov(V_k) = Q_k
% W_k zero mean uncorrelated gaussian, cov(W_k) = R_k
% V_k & W_j are uncorrelated for every k,j
%
% -------------------------------------------------------------------------
%
% Inputs:
% f = f_k
% config.Q = Q_k+1
% h = h_k
% y = y_k % measurements
% config.R = R_k
% del_f = gradient of f_k
% del_h = gradient of h_k
% x_hat = current state prediction
% P_hat = current error covariance (predicted)
% u = input information
% config.r = chi-square detector parameter
% CF_his = history function for DDE of state variable
% P_his = history function for DDE of covariance matrix
% groundtruth = groundtruth of the current state
% tk1,tk2 = time interval from the last sampling time tk1 to current
%           sampling time tk2
% config.tau = constant time delay
% config.OCSVM = use OCSVM or not
% psum = sum of the last N_ocsvm-1 normalized innovations
%
% -------------------------------------------------------------------------
%
% Outputs:
% x_next = next state prediction
% P_next = next error covariance (predicted)
% x_dgr = current state estimate
% P_dgr = current estimated error covariance
% p: p.chi = chi-square test statistics
%    p.innov = normalized innovation sequence
%    p.y_tilde = innovation of current time 
%    p.score = score of classifier
% K = Kalman Gain
% error = whether the current sensor reading is faulty (1) or not (0)
% 
% -------------------------------------------------------------------------
%

if isa(f,'function_handle') && isa(del_f,'function_handle') && isa(del_h,'function_handle') && isa(CF,'function_handle') && isa(dde_his,'function_handle') && isa(h,'function_handle')
    R = config.R;
    Q = config.Q;
    tau = config.tau;
    OCSVM = config.OCSVM;
    r = config.r;
    
    m = length(x_hat);                          % number of states

    f1 = @(tt,ss,ZZ) f(tt,ss,u,ZZ);

        
    error1 = 0;
    
    y_hat = h(x_hat);
    y_tilde = y - y_hat;                        % innovation    
    p.y_tilde = y_tilde;
    
    t = del_h(x_hat);                           % 1st Jacobian
    tmp = P_hat*t'; 
    
    S = t*tmp+R;                                % innovation covariance
    
    p.innov = abs(S.^(0.5)) \ y_tilde;

    K = tmp/(S+2*eps);                          % Kalman gain
    
    p.chi = y_tilde'/(S+2*eps)*y_tilde;         % chi-square statistics
    
    if(~OCSVM)
        
        if (p.chi >= r)
            error1 = 1;
            
            if(config.use_predict)
                x_dgr = x_hat;                  % if anomaly detected, use predict as estimate                
                P_dgr = P_hat;
            else
                x_dgr = groundtruth;
                t = del_h(groundtruth); 
                tmp = P_hat*t';
                S = t*tmp+R; 
                K = tmp/(S+2*eps); 
                P_dgr = P_hat - K*t*P_hat;
            end
        else 
            x_dgr = x_hat + K* y_tilde;
            P_dgr = P_hat - K*t*P_hat;
        end
    
    else
        OCSVM_r = config.OCSVM_threshold;
        
        pbar = norm((psum+p.innov)/config.N_ocsvm,1);
        
        if(pbar <= OCSVM_r(1))
            [~,score_1d] = predict(config.SVMModel1,p.innov');
        elseif((pbar <= OCSVM_r(2))&& (pbar > OCSVM_r(1)))
            [~,score_1d] = predict(config.SVMModel2,p.innov');
        elseif((pbar <= OCSVM_r(3))&& (pbar > OCSVM_r(2)))
            [~,score_1d] = predict(config.SVMModel3,p.innov');
        else
            [~,score_1d] = predict(config.SVMModel4,p.innov');
        end
        
        if(score_1d < 0)
            error1 = 1;
            
            if(config.use_predict)
                x_dgr = x_hat;                   % if anomaly detected, use predict as estimate 
                P_dgr = P_hat;
            else
                x_dgr = groundtruth;
                t = del_h(groundtruth); 
                tmp = P_hat*t';
                S = t*tmp+R; 
                K = tmp/(S+2*eps); 
                P_dgr = P_hat - K*t*P_hat;
            end
        else 
            x_dgr = x_hat + K* y_tilde;
            P_dgr = P_hat - K*t*P_hat;
        end
        p.score = score_1d;
    end
    
    y_residual = y - h(x_dgr);                   % residual between the actual measurement and its estimated value
    p.residual = y_residual;
    
    x_der = @(s) CF(s,u);
    P_der = @(s,P) reshape(P*del_f(s,u) + del_f(s,u)* P' + Q, [], 1);
    
    sys_his = @(tt) dde_his(tt,[x_dgr;reshape(P_dgr,[],1)]);
    
    if(tau>0)
        dde_sys = @(t,sys_state, Z) dde_ss(t, sys_state, Z, x_der, P_der);
        sol_sys = dde23(dde_sys,tau,sys_his,[tk1,tk2]);    %solve DDE of state & covariance 
    else
        ode_sys = @(t,s)  ode_ss(t,s,x_der, P_der);
        sol_sys = ode15s(ode_sys,[tk1,tk2],sys_his(1));
    end
    
    P_next = reshape(squeeze(sol_sys.y(m+1:end,end)), m, m);
    
    %======================================================================
    %===Force covariance matrix symmetric and positive on diag elements====
    %======================================================================
    for k = 1:m
    	P_next(k,k) = abs(P_next(k,k));
    end
    P_next = (P_next + P_next')*0.5;
    %======================================================================
    %======================================================================
    %======================================================================
    x_next = squeeze(sol_sys.y(1:m,end));
    
    p.RMSE = sqrt(mean((groundtruth - x_dgr).^2));  % Root Mean Squared Error
    
else
    error('f, h, del_f, del_h, CF_his and P_his should be function handles')
    return
end
end


function dde_sys = dde_ss(t, s, Z, x_der, P_der)
%x_der,P_der are function handles

xlag = Z(:,1);

dde_sys = [ Eselect(x_der([xlag(1);xlag(2)]), 1);
            Eselect(x_der([xlag(1);xlag(2)]), 2);
            Eselect(P_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 1);
            Eselect(P_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 2);
            Eselect(P_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 3);
            Eselect(P_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 4)];
end

function ode_sys = ode_ss(t,Z,x_der, P_der)
x = Z(:,1);

ode_sys = [ Eselect(x_der([x(1);x(2)]), 1);
            Eselect(x_der([x(1);x(2)]), 2);
            Eselect(P_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 1);
            Eselect(P_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 2);
            Eselect(P_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 3);
            Eselect(P_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 4)];
end
