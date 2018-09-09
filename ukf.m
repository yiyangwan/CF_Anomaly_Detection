function [x_next,P_next,x_dgr,P_dgr,p,K,error1] = ukf(f,h,y,x_hat,P_hat,u,groundtruth,CF,dde_his,tk1,tk2,config,psum) 
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

if isa(f,'function_handle') && isa(CF,'function_handle') && isa(dde_his,'function_handle') && isa(h,'function_handle')
    R       = config.R;
    Q       = config.Q;
    tau     = config.tau;
    OCSVM   = config.OCSVM;
    r       = config.r;
    
    x = x_hat.x;                                %state prediction
    X = x_hat.X;                                %sigma points prediction
    
    L       = numel(x);                         %numer of states
    m       = numel(y);                         %numer of measurements
    alpha   = config.alpha;                     %default, tunable
    ki      = config.ki;                        %default, tunable
    beta    = config.beta;                      %default, tunable
    
    X2      = X-x(:,ones(1,2*L+1));             %residuals of sigma points

    
    lambda  =alpha^2*(L+ki)-L;                  %scaling factor
    c       =L+lambda;                          %scaling factor
    Wm      =[lambda/c 0.5/c+zeros(1,2*L)]';    %weights for means
    Wc      =Wm;
    Wc(1)   =Wc(1)+(1-alpha^2+beta);            %weights for covariance
    c       =sqrt(c);
    
    W       = ( eye(2*L+1) - Wm*ones(1,(2*L+1)) ) * diag(Wc) ...
        * ( eye(2*L+1) - Wm*ones(1,(2*L+1)) )'; %weight matrix used for solving DDE
    
    
    error1  = 0;                                %fault indicator
          
    [y1,~,P2,Y2]    =ut(h,X,Wm,Wc,m,R);         %unscented transformation of measurments
    
    p.y_tilde       = y-y1;                     %innovation
    p.innov         = abs(P2.^(0.5))\p.y_tilde; %normalized innovation sequence
    
    P12             =X2*diag(Wc)*Y2';       	%transformed cross-covariance
    K               =P12 / (P2+2*eps);      	%kalman gain
    
    p.chi = p.y_tilde'/(P2+2*eps+R)*p.y_tilde;  %chi-square statistics
       
    if(~OCSVM)
        if (p.chi >= r)
            error1 = 1;
            if(config.use_predict)
                x_dgr = x;                      %if anomaly detected, use predict as estimate 
                P_dgr = P_hat;
            else
                x_dgr = groundtruth;
                P_dgr = P_hat - K*P12';
            end
        else 
            x_dgr = x + K * p.y_tilde;
            P_dgr = P_hat - K*P12';
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
                P_dgr = P_hat - K*P12';
            end
            
        else 
            x_dgr = x + K* p.y_tilde;
            P_dgr = P_hat - K*P12';
        end
            p.score = score_1d;
    end        
        
        for k = 1:L
            P_dgr(k,k) = abs(P_dgr(k,k));
        end
        
        p.residual = y - h(x_dgr);                      %residual between the actual measurement and its estimated value
                         
        CF1 = @(s) CF(s,u); 
        CF_Sigma = @(X) CF_X(X,CF1);                    %motion model for sigma points matrix
        
        x_der  = @(s,P) CF_Sigma(sigmas(s,P,c)) * Wm;
        
        P_der = @(s,P) reshape(sigmas(s,P,c) * W * CF_Sigma(sigmas(s,P,c))' + CF_Sigma(sigmas(s,P,c)) * W * sigmas(s,P,c)' + Q,[],1);
        
        sys_his = @(tt) dde_his(tt,[x_dgr;reshape(P_dgr,[],1)]);
        
        if(tau>0)
            dde_sys = @(t,sys_state,Z) dde_ss(t,sys_state,Z,x_der,P_der);
            sol_sys = dde23(dde_sys,tau,sys_his,[tk1,tk2]);    %solve DDE of state & covariance 
        else
            ode_sys = @(t,s)  ode_ss(t,s,x_der, P_der);
            sol_sys = ode45(ode_sys,[tk1,tk2],sys_his(1));
        end
        
        P_next = reshape(squeeze(sol_sys.y(L+1:end,end)), L,L);

     	%======================================================================
        %===Force covariance matrix symmetric and positive on diag elements====
        %======================================================================
        for k = 1:L
            P_next(k,k) = abs(P_next(k,k));
        end
        P_next = (P_next + P_next')*0.5;
        %======================================================================
        %======================================================================
        %======================================================================
        x_next.x = squeeze(sol_sys.y(1:L,end));
        x_next.X = sigmas(x_next.x,P_next,c);       
    
        p.RMSE = sqrt(mean((groundtruth - x_dgr).^2));  %Root Mean Squared Error
            
else
    error('f, h, CF and dde_his should be function handles')
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

function cf = CF_X(X,CF)
%CF is a function handle @(s)!
C = num2cell(X, 1);                     %Collect the columns into cells
cf = cell2mat(cellfun(CF, C,'un',0));   %A 2-by-(2L+1) vector
end

function dde_sys = dde_ss(t, s, Z, x_der, P_der)
%x_der,P_der are function handles
xlag = Z(:,1);

dde_sys = [ Eselect(x_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 1);
            Eselect(x_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 2);
            Eselect(P_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 1);
            Eselect(P_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 2);
            Eselect(P_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 3);
            Eselect(P_der([xlag(1);xlag(2)], [xlag(3),xlag(5);xlag(4),xlag(6)]), 4) ];
end

function ode_sys = ode_ss(t, Z, x_der, P_der)
x = Z(:,1);

ode_sys = [ Eselect(x_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 1);
            Eselect(x_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 2);
            Eselect(P_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 1);
            Eselect(P_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 2);
            Eselect(P_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 3);
            Eselect(P_der([x(1);x(2)], [x(3),x(5);x(4),x(6)]), 4)];
end
