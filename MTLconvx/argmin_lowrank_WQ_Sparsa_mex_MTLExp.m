%% FUNCTION argmin
%  solve model w and Q. 
%  Use SS: a specified structure sample to vectorization Q and accelerate
%  the speed.
%
%% OBJECTIVE
%  min_{W,Z1,Z2,Z3} \sum_{t= 1}^{K} \sum_{i=1}^{m_t} 1/2 ||xi*wt +  xi^T( \sum_{j=1}^3 Zjt )xi - yti||_2^{2} + \lambda ||W||_{tr} + \mu \sum_{j=1}^3||\vZj(j)||_{tr}
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

function [W,Q] = argmin_lowrank_WQ_Sparsa_mex_MTLExp(X, Y, W, Q, rho1, rho2,FISTA_options)
 
[d,K] = size(W);


% Vectorization of W and Q. 
z = vectorization(W,Q);
% disp('finish vec')
z_func = @(x)gradient(x, X, Y);

[z_new_vect, ~, output] = pnopt_sparsa(z_func, project(rho1, rho2, d, K), z,FISTA_options);

% f_x = output.trace.f_x;
% disp('finish de-vec')
[W,Q] = de_vectorization(z_new_vect,d,K);
% plot(f_x);
Q = Q{1} + Q{2} + Q{3}; 
end




function [f, grad_z_vec] = gradient(z_vec, X, Y)

K = size(Y,2);
d = size(X,2);

[W,Q] = de_vectorization(z_vec,d,K);

[grad_W,grad_Q,f] = gradFuncMTLExpConvx(X,Y,W,Q,d,K);

grad_Q_cell = cell(3,1);
for j = 1:3 grad_Q_cell{j} = grad_Q; end

grad_z_vec = vectorization(grad_W, grad_Q_cell);



end





function op = project(rho1, rho2, d, K) % this part deal with the non smooth part.
op = tfocs_prox( @f, @(x,t)prox_f(rho1,rho2,x,t) , 'vector' ); % Allow vector stepsizes


    %estimator = cell(3,1);
    
    function v = f(x)
        %  function value of non-smooth part
        
        
        [W,Q] = de_vectorization(x,d,K);
        f_W = trace_norm(W) * rho1;
        
        % calculate nonsmooth function value of tensor Q. 
        f_Q = 0;
        for i = 1:3
            Qi = unfolding(Q{i},i);
            f_Q = f_Q + trace_norm(Qi)*rho2;
        end

        v   = f_W + f_Q ;    
    end
      
    function x = prox_f(rho1,rho2,v,t)
        
        % Recover weight matrix w and interacting tensor Q.
        N = d*d*K;
        v_w = v(1:d*K);
        v_Q = zeros(N*3,1);
        
        % this projection calculates
        % argmin_z = 1/2\|z-v\|_2^2 + beta \|z\|_tr
        % z: solution
        W = reshape(v_w,d,K);
        W = prox_tr(W,rho1*t);
        w = W(:);
        
        Q = cell(3,1);
        for i = 1:3     % project each tensor to the correspond low rank unfolding
            Q{i}     = reshape(v((1+(i-1)*d)*d*K +1:(1+(i-1)*d+d)*d*K),d,d,K);
            Qt       = unfolding(Q{i},i);
            Qt_new   = prox_tr(Qt,rho2*t);
            Q{i}     = folding(Qt_new,size(Q{i}),i);
        end
                
        for j = 1:3     % vectorize 3 tensors.
            v_Q((j-1)*N+1:j*N,1) = reshape(Q{i},N,1);
        end
        
        q = v_Q; 
        x = [w',q']';
           
    end
end