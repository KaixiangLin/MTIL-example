%% FUNCTION block_main
%  Using block coordinate descent to solve non convex formulation.
% 
%% OBJECTIVE
%  \sum_t=1,..K \sum_j=1...mt ||xtj'*Wt + xtj'*B*q_t*B'*xtj - ytj||_F^2 +
%  rho1*||W||_tr
%               
%  
%% INPUT
%   Q: d * d * K interact feature matrix 
%   r: rank. design the rank of tensor. 
%
%% OUTPUT
%   B: model: d * r.   common basis matrix for all tasks.
%   
%% Code starts here
 

function [W,Q,B,q,F_val] = LRW_LRnQreg_block_main_MTL_cell(X, Y, W, B, q,  parameters)
 
maxIter = parameters.maxIter;
maintol = parameters.maintol;
lambdaB = parameters.lambdaB;
lambdaq = parameters.lambdaq;
FISTA_options = parameters.FISTA_options;
rho1 = parameters.lambdaW;
r    = parameters.rank;
iter = 0;

[d,K] = size(W);

% Initialization  
% f_k_prev = 0;
F_val = [];

while true
    
    [W_new, funcval] = argmin_LRW_sparsa_MTL_cell(X, Y, W, q, B, d, K, r, rho1,FISTA_options);
    
 
    
    [q_new, funcval] = argmin_qreg_sparsa_MTL_cell(X, Y, W_new, q,     B, d, K, r,lambdaq, FISTA_options);
 
    
    [B_new, funcval] = argmin_Breg_sparsa_MTL_cell(X, Y, W_new, q_new, B, d, K, r, lambdaB, FISTA_options);
     funcval = obj_func(X,Y,W,B,q,d,K, r,rho1,lambdaB,lambdaq);
     F_val = [F_val, funcval];
    
 
     
%     if (iter > maxIter | abs(f_k_prev - f_k_curr) < maintol)
    if (iter > maxIter)
        W = W_new;
        q = q_new;
        B = B_new;
        break;
%     elseif (iter > 2 && abs(F_val(end) - F_val(end-1))< abs(maintol))
% % %         disp('Nov:break stop');
%         W = W_new;
%         q = q_new;
%         B = B_new;
% % 
%         break;
    else
        
        W = W_new;
        q = q_new;
        B = B_new;
%         f_k_prev = f_k_curr;
        iter = iter + 1;
%         if mod(iter,30)==0
%            disp(sprintf('---the %d th iteration------',iter));
%         end
    end
    
end

Q = zeros(d,d,K);
for i = 1:K
    Q(:,:,i) = B*q(:,:,i)*B';
end

% plot(F_val,'-o');

end


function f = obj_func(X,Y,W,B,q,d ,K, r,rho1,lambdaB,lambdaq)
%  f = 0;
%  for i = 1:K
%     Xt = X{i};
%     mt = size(Xt,1);
%     Wt = W(:,i);
%     qt = q(:,:,i); % r*r
%     Qt = B*qt*B';
%     Yt = Y{i};
% 
%     for j = 1:mt
%        Xtj = Xt(j,:);  % 1 * d
%        ZZ = Xtj*Wt - Yt(j);
%        Xtemp = (Xtj*Qt*Xtj' + ZZ);
% 
%        f = f + 1/2*Xtemp*Xtemp;
%     end
%  end

 f = smooth_funcvalueNcvreg(X,Y,W,q,B,d,K,r,lambdaB,lambdaq);
 f = f + trace_norm(W) * rho1;
end

