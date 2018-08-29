function X=sigmas(x,P,c)
%Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       c: coefficient
%Output:
%       X: Sigma points
[cholesky, ind] = chol(P);
if(issymmetric(P))
    if(~ind)
        A = c*cholesky';
    end
    
else %force matrix P to be symmetric and PD such that Cholesky factorization exists
    for i = 1:size(P,1)-1
        for j = i+1:size(P,2)
            P(i,j) = 0.5*(P(j,i)+P(i,j));
            P(j,i) = P(i,j);
        end
    end
    P = topdm(P);                           %force P to be a PD matrix
    A = c*chol(P)';
end
Y = x(:,ones(1,numel(x)));
X = [x Y+A Y-A]; 
end