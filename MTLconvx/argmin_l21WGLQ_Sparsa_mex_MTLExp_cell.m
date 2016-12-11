%% FUNCTION argmin
%  solve model w and Q. 
%  Feb 10th
%   
%
%% OBJECTIVE
%  
%               
%  
%% INPUT
%   X: {K * 1}  - Sample cell for K tasks. X{t}: mt * d matrix. mt is the number of samples for task 1. 
%                 d is the dimension of each sample. NOTICE: d is same for all tasks.
%   Y: {K * 1}  - Lable cell for K tasks. Y{t}: mt * 1 vector. mt is the number of samples for task 1. 
%   rho1: sprasity controlling parameter for w
%   rho2: Low rank regularization parameter for tensor Q.
%   W:    d * K vector from last step.
%   Q: {K * 1} cell Q{1} is d * d * K interact feature matrix for this task.
%
%% OUTPUT
%   W: model: d * 1.     - Weight matrix for all tasks.

%% Code starts here

function [W,Q,f_x] = argmin_l21WGLQ_Sparsa_mex_MTLExp_cell(X, Y, W, Q, rho1, rho2,FISTA_options)
 

d = size(X{1},2);
K = size(X,1);
% Vectorization of W and Q. 
z = vectorizationMatWTensorQ(W,Q);
% disp('finish vec')
z_func = @(x)gradient(x, X, Y);

[z_new_vect, ~, output] = pnopt_sparsa(z_func, project(rho1, rho2, d, K), z,FISTA_options);

f_x = output.trace.f_x;
% disp('finish de-vec')
[W,Q] = de_vectorizationMatWTensorQ(z_new_vect,d,K);
% plot(f_x);
 
end




function [f, grad_z_vec] = gradient(z_vec, X, Y)

K = size(X,1);
d = size(X{1},2);

[W,Q] = de_vectorizationMatWTensorQ(z_vec,d,K);

[grad_W,grad_Q,f] = gl_LowrankgradFunval(X,Y,W,Q,d,K);

 

grad_z_vec = vectorizationMatWTensorQ(grad_W, grad_Q);

end


function z = vectorizationMatWTensorQ(W, Q) % checked.

[d, K] = size(W);

z = zeros((1+d*K)*d,1);

 
dK = d*K;
N  = dK*d;
z(1:dK) = reshape(W,dK,1);
z(dK+1:dK+N) = reshape(Q,N,1);

end 

function [W,Q] = de_vectorizationMatWTensorQ(z,d,K) % checked.

W = reshape(z(1:d*K),d,K);
Q = reshape(z(d*K +1:(1+d)*d*K),d,d,K);


end 




function op = project(rho1, rho2, d, K) % this part deal with the non smooth part.
op = tfocs_prox( @f, @(x,t)prox_f(rho1,rho2,x,t) , 'vector' ); % Allow vector stepsizes


    %estimator = cell(3,1);
    
    function v = f(x)
        %  function value of non-smooth part
        
        
        [W,Q] = de_vectorizationMatWTensorQ(x,d,K);
        
        f_W = sum(sqrt(sum(W.^2))) * rho1;
        % calculate nonsmooth function value of tensor Q. 
      
        Qunfolding   = unfolding(Q,3); % 139*(28*28)
        f_Q = sum(sqrt(sum(Qunfolding.^2)));

        v   = f_W + f_Q * rho2;    
    end
      
    function x = prox_f(rho1,rho2,v,t)
        
        % Recover weight matrix w and interacting tensor Q.
        
        [W,Q] = de_vectorizationMatWTensorQ(v,d,K);
        
        % this projection calculates
        % argmin_z = 1/2\|z-v\|_2^2 + beta \|z\|_tr
        % z: solution
        % W = prox_tr(W,rho1*t);
        W = repmat(max(0, 1 - rho1*t./sqrt(sum(W.^2,2))),1,size(W,2)).*W;
        
        w = W(:);
        Qunfolding   = unfolding(Q,3);
        D = Qunfolding';
        Q_shrink = repmat(max(0, 1 - rho2*t./sqrt(sum(D.^2,2))),1,size(D,2)).*D;
        
        Q_new = folding(Q_shrink',[d,d,K],3);
 
        q = reshape(Q_new,d*d*K,1);  % order: first slice: 1st col,2nd col,...;second slice:1st col, 2nd col.
        
        x = [w',q']';
           
    end
end

