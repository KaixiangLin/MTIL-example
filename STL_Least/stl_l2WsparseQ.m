%% FUNCTION argmin
%  solve model w and Q. 
%  single task learning with feature interaction. using a sparse symmetric
%  matrix Q to capture the feature interaction.
%   
%
%% OBJECTIVE
%  Single task learning with feature interaction in the ExpNotes.
%               
%  
%% INPUT
%   X: mt *d   -  mt * d matrix. mt is the number of samples for task 1. 
%                 d is the dimension of each sample. NOTICE: d is same for all tasks.
%   Y: {K * 1}  - Lable cell for K tasks. Y{t}: mt * 1 vector. mt is the number of samples for task 1. 
%   rho1: sprasity controlling parameter for w
%   rho2: Low rank regularization parameter for tensor Q.
%   w:    d * 1 vector ini.
%   Q:    d * d interact feature matrix symmetric.
%
%% OUTPUT
%   W: model: d * 1.     - Weight matrix for all tasks.
    
%% Code starts here


function [w, Q, f_x] = stl_l2WsparseQ(X,Y,w,Q, rho1,rho2, FISTA_options)
 
[d] = size(w,1);


% Vectorization of W and Q. 
z = vectorization(w,Q);
% disp('finish vec')
z_func = @(x)gradient(x, X, Y,rho1);

[z_new_vect, ~, output] = pnopt_sparsa(z_func, project(rho2, d), z,FISTA_options);

f_x = output.trace.f_x;
% plot(f_x);

[w,Q] = de_vectorization(z_new_vect,d);

  
end

function z = vectorization(w,Q)
% reshape a d*1 vector w and d*d matrix to d*d+d vector z

    d = size(Q,1);
    z = zeros(d*(d+1),1);
    z(1:d) = w;
    z(d+1:end) = reshape(Q,d*d,1);

end

function [w,Q] = de_vectorization(z,d)
% reshape a d*d+d vector z to a d*1 vector w and d*d matrix Q
w = z(1:d,1);
Q = reshape(z(d+1:end,1),d,d);

end

function [f, grad_z_vec] = gradient(z_vec, X, Y, rho1)

[m, d] = size(X);

[w,Q] = de_vectorization(z_vec,d);

% grad_w = zeros(d,1);
% grad_Q = zeros(d,d);
% f = 0;
% for i = 1:m
%     xi = X(i,:);
%     
%     Xtemp = (xi*w + xi*Q*xi' - Y(i));
% 
%     grad_w = grad_w + xi'*Xtemp; 
%     
%     grad_Q = grad_Q + (xi'*Xtemp)*xi;
%     
%     f = f + Xtemp*Xtemp;
% end
% 
% grad_w = grad_w+ rho1 * w;
% f = (f + rho1 * sum(w.^2))/2;

[grad_w, grad_Q, f] = norm2WsparsesymQ(X,Y,w,Q,d,rho1);

grad_z_vec = vectorization(grad_w, grad_Q);

end





function op = project(rho2, d) % this part deal with the non smooth part.
op = tfocs_prox( @f, @(x,t)prox_f(rho2,x,t) , 'vector' ); % Allow vector stepsizes


    %estimator = cell(3,1);
    
    function v = f(x)
        %  function value of non-smooth part
        
        
        [w,Q] = de_vectorization(x,d);

        v   = rho2*norm(Q(:),1) ;    
    end
      
    function x = prox_f(rho2,v,t)
        
        % Recover weight matrix w and interacting tensor Q.
        
        
        x = zeros((d+1)*d,1);
        
        x(1:d,1) = v(1:d,1);
        
        q = v(d+1:end,1);
        
        x(d+1:end,1) = sign(q).*max(0,abs(q)- rho2*t);
        
        [w,Q] = de_vectorization(x,d);
           
    end
end

