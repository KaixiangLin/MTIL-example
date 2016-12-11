%% FUNCTION argmin
%  solve model w and Q. 
%
%
%% OBJECTIVE
%  
%               
%  
%% INPUT
%   X: m * d    - Sample matrix for K tasks.  NOTICE: m, d is same for all tasks.
%   Y: {K * 1}  - Lable cell for K tasks. Y{t}: mt * 1 vector. mt is the number of samples for task 1. 
%   rho1: sprasity controlling parameter for w
%   W:    d * K vector from last step.
%
%% OUTPUT
%   W: model: d * 1.     - Weight matrix for all tasks.

%% Code starts here

function [W,f] = argmin_lowrank_W_exp_cell(X, Y, W, rho1,FISTA_options)
% addpath('../')

[d,K] = size(W);

% Vectorization of W 
z = reshape(W,d*K,1);

z_func = @(x)gradient(x, X, Y);

[z_new_vect,~,output] = pnopt_sparsa(z_func, project(rho1, d, K), z,FISTA_options);
f = output.trace.f_x; 
%de_vectorization 
W = reshape(z_new_vect,d,K);
end




function [f, grad_z_vec] = gradient(z_vec, X, Y)

K = size(X,1);
d = size(X{1},2);


W = reshape(z_vec,d,K);


% Calculate gradient of W and Q. 
grad_W = zeros(size(W));
 
f = 0;

for i = 1:K
 
    Xt = X{i}; % mt * d
    Wt = W(:,i);
    Yt = Y{i};
   
    Ptemp   = Xt*Wt - Yt;  % temp result for gradient and objective function value.
    
    grad_W(:,i) = grad_W(:,i) + Xt'*Ptemp;
    

    % Calculate smooth function value
    f = f + sum(Ptemp.^2);
end

 
% Vectorization of W 
grad_z_vec = reshape(grad_W,d*K,1);

f = f/2;

end





function op = project(rho1, d, K) % this part deal with the non smooth part.
op = tfocs_prox( @f, @(x,t)prox_f(rho1,x,t) , 'vector' ); % Allow vector stepsizes


 
    
    function v = f(x)
        %  function value of non-smooth part
        
        %de_vectorization 
        W = reshape(x,d,K);
        f_W = trace_norm(W) * rho1;
        v   = f_W ;    
    end
      
    function x = prox_f(rho1,v,t)
        
        % this projection calculates
        % argmin_z = 1/2\|z-v\|_2^2 + beta \|z\|_tr
        % z: solution
        W = reshape(v,d,K);
        X = prox_tr(W,rho1*t);
        x = reshape(X,d*K,1);
    end
end