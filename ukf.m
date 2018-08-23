function [x_next,P_next,x_dgr,P_dgr,p,K,error1] = ukf(f,h,y,x_hat,P_hat,u,groundtruth,CF_his,P_his,tk1,tk2,config,psum) 
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
%       x_hat.X = current sigma points predictioin
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

if isa(f,'function_handle') && isa(P_his,'function_handle') && isa(CF_his,'function_handle') && isa(h,'function_handle')
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

else
    error('f, h, CF_his and P_his should be function handles')
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