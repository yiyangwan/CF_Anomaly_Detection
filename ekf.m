function [x_next,P_next,x_dgr,P_dgr,p,K,error1] = ekf(f,h,y,del_f,del_h,x_hat,P_hat,u,groundtruth,CF_his,P_his,tk1,tk2,config,psum)
% Extended Kalman filter
%
% -------------------------------------------------------------------------
%
% State space model is
% X_k+1 = f_k(X_k) + V_k+1   -->  state update
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

if isa(f,'function_handle') && isa(del_f,'function_handle') && isa(del_h,'function_handle') && isa(P_his,'function_handle') && isa(CF_his,'function_handle') && isa(h,'function_handle')
    R = config.R;
    Q = config.Q;
    tau = config.tau;
    OCSVM = config.OCSVM;
    r = config.r;
    
    f1 = @(tt,ss,ZZ) f(tt,ss,u,ZZ);
    
    error1 = 0;
    
    y_hat = h(x_hat);
    y_tilde = y - y_hat;        % innovation
    p.y_tilde = y_tilde;
    
    t = del_h(x_hat);           % 1st Jacobian
    tmp = P_hat*t'; 
    
    S = t*tmp+R;               % innovation covariance
    
    p.innov = abs(S.^(0.5)) \ y_tilde;

    K = tmp/(S+eps);            % Kalman gain
    
    p.chi = y_tilde'/(S+eps)*y_tilde; % chi-square statistics
    
    if(~OCSVM)
        
        if (p.chi >= r)
            error1 = 1;
            x_dgr = groundtruth;
        else 
            x_dgr = x_hat + K* y_tilde;
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
            x_dgr = groundtruth;
        else 
            x_dgr = x_hat + K* y_tilde;
        end
        p.score = score_1d;
    end

    CF_his1 = @(tt) CF_his(tt,x_dgr);  % store current state estimate to compute the next prediction     
%     x_next = f(x_dgr);
    sol_x = dde23(f1,tau,CF_his1,[tk1,tk2]); % solve DDE of state variable 
    x_next = sol_x.y(:,end);
    
    P_dgr = P_hat - K*t*P_hat;
%     P_next = p* P_dgr* p' + Q;
    f_del1 = @(tt,P,Z) P_d(tt,P,Z,del_f(x_dgr,u),Q);
    P_his_1 = @(tt) reshape(P_his(tt,P_dgr),[],1); % store current covariance estimate to compute the next prediction
    sol_P = dde23(f_del1,tau,P_his_1,[tk1,tk2]); % solve DDE of covariance matrix P
    P_next = reshape(sol_P.y(:,end),2,2);
    
else
    error('f, h, del_f, del_h, CF_his and P_his should be function handles')
    return
end
end

function p = P_d(tt,P,Z,del_f,Q)
plag1 = Z(:,1);

a = del_f(1,1); b = del_f(1,2); c = del_f(2,1); d = del_f(2,2);

p = [a*plag1(1)+b*plag1(3); a*plag1(2)+b*plag1(4);...
    c*plag1(1)+d*plag1(3); c*plag1(2)+d*plag1(4)] + ...
    [a*plag1(1)+b*plag1(2); c*plag1(1)+d*plag1(2); ...
    a*plag1(3) + b*plag1(4); c*plag1(3)+d*plag1(4)] + reshape(Q,[],1);
end