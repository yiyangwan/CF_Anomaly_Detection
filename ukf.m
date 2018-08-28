function [x_next,P_next,x_dgr,P_dgr,p,K,error1] = ukf(f,h,y,x_hat,P_hat,u,groundtruth,CF,CF_his,P_his,tk1,tk2,config,psum) 
% UKF   Unscented Kalman Filter for nonlinear dynamic systems
% -------------------------------------------------------------------------
%
% Inputs:
% f = f_k
% config.Q = Q_k+1
% h = h_k
% y = y_k % measurements
% config.R = R_k
% x_hat: 
%       x_hat.x = current state prediction
%       x_hat.X = current sigma points prediction
% P_hat = current error covariance (predicted)
% u = input information
% config.r = chi-square detector parameter
% CF = Motion model
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
% x_next:
%        x_next.x = next state prediction
%        x_next.X = next sigma points prediction
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

if isa(f,'function_handle') && isa(CF,'function_handle') && isa(P_his,'function_handle') && isa(CF_his,'function_handle') && isa(h,'function_handle')
    R = config.R;
    Q = config.Q;
    tau = config.tau;
    OCSVM = config.OCSVM;
    r = config.r;
    
    x = x_hat.x;                                %state prediction
    X = x_hat.X;                                %sigma points prediction
    
    L = numel(x);                               %numer of states
    m = numel(y);                               %numer of measurements
    alpha = config.alpha;                       %default, tunable
    ki = config.ki;                             %default, tunable
    beta= config.beta;                          %default, tunable
    
    X2 = X-x(:,ones(1,L));                      %residuals of sigma points

    
    lambda=alpha^2*(L+ki)-L;                    %scaling factor
    c=L+lambda;                                 %scaling factor
    Wm=[lambda/c 0.5/c+zeros(1,2*L)];           %weights for means
    Wc=Wm;
    Wc(1)=Wc(1)+(1-alpha^2+beta);               %weights for covariance
    c=sqrt(c);
    
    error1 = 0;                                 %fault indicator
          
    [y1,Y1,P2,Y2]=ut(h,X,Wm,Wc,m,R);            %unscented transformation of measurments
    
    p.y_tilde = y-y1;                           %innovation
    p.innov = abs(P2.^(0.5)) \ p.y_tilde;       %normalized innovation sequence
    
    P12=X2*diag(Wc)*Y2';                        %transformed cross-covariance
    K=P12 / (P2+2*eps);                         %kalman gain
    
    p.chi = p.y_tilde'/(P2+2*eps+R)*p.y_tilde;  %chi-square statistics
    
    f1 = @(tt,ss,ZZ) f(tt,ss,u,ZZ);
    
    P=P1-K*P12';                                %covariance update

    if(~OCSVM)
        if (p.chi >= r)
            error1 = 1;
            if(config.use_predict)
                x_dgr = x;                      %if anomaly detected, use predict as estimate 
                P_dgr = P_hat;
            else
                x_dgr = groundtruth;
                P_dgr = P_hat - K*t*P_hat;
            end
        else 
            x_dgr = x + K * p.y_tilde;
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
                x_dgr = x;
                P_dgr = P_hat;
            else
                x_dgr = groundtruth;
                P_dgr = P_hat - K*t*P_hat;
            end
            
        else 
            x_dgr = x + K* y_tilde;
            P_dgr = P_hat - K*t*P_hat;
        end
            p.score = score_1d;
    end        
        
        y_residual = y - h(x_dgr);                  %residual between the actual measurement and its estimated value
        p.residual = y_residual;
        
        
%         X_dgr = sigmas(x_dgr,P_dgr,c);              %sigma points of current estimation
%         X_next = zeros(size(X_dgr));                %initial predicted sigma points
%         x_next.x = zeros(size(x));
        
%         for i = 1:(2*L+1)
%             
%             CF_his1 = @(tt) CF_his(tt,X_dgr(:,i));    %store current state estimate to compute the next prediction
%             sol_X = dde23(f1,tau,CF_his1,[tk1,tk2]);  %solve DDE of state variable
%             x_next.X(:,i) = sol_X.y(:,end);
%             x_next.x = x_next.x + Wm(i) * x_next.X(:,i);
%                        
%         end
        
        CF1 = @(s) CF(s,u); 
        CF_Sigma = @(X) CF_X(X,CF1);                    %motion model for sigma points matrix
        
        x_der  = @(s) CF_Sigma(sigmas(s,P_dgr,c)) * Wm;
        
        P_der = @(s) reshape(sigmas(s,P_dgr,c)* W * x_der(s,P_dgr,c)' + x_der(s,P_dgr,c) *W * sigmas(s,P_dgr,c)' + Q,[],1);
        
        dde_sys = @(t,s,Z) dde_ss(t,s,Z,x_der,P_der);
        
%         f_del1 = @(tt,P,Z) P_d(tt,P,Z,CF_Sigma(X_dgr(:,i),u),Q);
%         P_his_1 = @(tt) reshape(P_his(tt,P_dgr),[],1);  %store current covariance estimate to compute the next prediction
        sol_P = dde23(dde_sys,tau,P_his_1,[tk1,tk2]);    %solve DDE of covariance matrix P **************************I AM HERE!!!!*******************
        P_next = reshape(sol_P.y(:,end),2,2); 
        
        
             
        
    
        p.RMSE = sqrt(mean((groundtruth - x_dgr).^2));  %Root Mean Squared Error
     
        
else
    error('f, h, CF, CF_his and P_his should be function handles')
    return
end
end

function [y,Y,P,Y1]=ut(f,X,Wm,Wc,n,R)
%Unscented Transformation
%Input:
%        f: nonlinear map
%        X: sigma points
%       Wm: weights for mean
%       Wc: weights for covraiance
%        n: numer of outputs of f
%        R: additive covariance
%Output:
%        y: transformed mean
%        Y: transformed smapling points
%        P: transformed covariance
%       Y1: transformed deviations

L=size(X,2);
y=zeros(n,1);
Y=zeros(n,L);
for k=1:L                   
    Y(:,k)=f(X(:,k));       
    y=y+Wm(k)*Y(:,k);       
end
Y1=Y-y(:,ones(1,L));
P=Y1*diag(Wc)*Y1'+R;     
end

function X=sigmas(x,P,c)
%Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       c: coefficient
%Output:
%       X: Sigma points

A = c*chol(P)';
Y = x(:,ones(1,numel(x)));
X = [x Y+A Y-A]; 
end

function p = P_d(tt,P,Z,f,Q)
plag1 = Z(:,1);

a = f(1,1); b = f(1,2); c = f(2,1); d = f(2,2);

p = [a*plag1(1)+b*plag1(3); a*plag1(2)+b*plag1(4);...
    c*plag1(1)+d*plag1(3); c*plag1(2)+d*plag1(4)] + ...
    [a*plag1(1)+b*plag1(2); c*plag1(1)+d*plag1(2); ...
    a*plag1(3) + b*plag1(4); c*plag1(3)+d*plag1(4)] + reshape(Q,[],1);
end

function cf = CF_X(X,CF)
% CF is a function handle @(s)!
C = num2cell(X, 1);                     %Collect the columns into cells
cf = cellfun(CF, C);          %A 2-by-(2L+1) vector
end

function dde_sys = dde_ss(t,s,Z,x_der,P_der)
%x_der,P_der are function handles
xlag = Z(:,1);

dde_sys = [select(x_der([xlag(1);xlag(2)]), 1);
            select(x_der([xlag(1);xlag(2)]), 2);
            select(P_der([xlag(1);xlag(2)]), 1);
            select(P_der([xlag(1);xlag(2)]), 2);
            select(P_der([xlag(1);xlag(2)]), 3);
            select(P_der([xlag(1);xlag(2)]), 4)];

end


% function cf2 = CF_X2(t,X,Z,f1)
% % f1 is a function handle @(tt,ss,ZZ)!
% C = num2cell(X, 1);                      %Collect the columns into cells
% f11 = @(ss) f1(t,ss,Z);
% cf2 = cellfun(f11, C);
% end