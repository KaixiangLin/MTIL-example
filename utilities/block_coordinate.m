%% FUNCTION argmin
%  Low rank decomposition of tensor Q. 
%
%% OBJECTIVE
%  \sum_t=1,..K ||Q_t - B*q_t*B'||_F^2
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

function [B,q,F] = block_coordinate(Q, r,FISTA_options)
    tol     = 1e-6;
    B_tol   = 1e-3;
    maxIter = 100;
    iter    = 0;

    K = size(Q,3);
    d = size(Q,1);
    
    B = rand(d,r);
    q = rand(r,r,K);
    F = [];
    while(iter < maxIter)
        
        q_new = zeros(r,r,K);
        
        invB  = inv(B'*B);
        % argmin Q for the objective function.
        for i = 1:K
           q_new(:,:,i) = invB*B'*Q(:,:,i)*B*invB;     %gradient checked. 
%              q_new(:,:,i) =      B'*Q(:,:,i)*B
        end
  
        
        f_v = func(B,Q,q_new);
        F = [F,f_v];
        
        % argmin B for the objective function.
        B_new = argmin_B_FISTA(Q,B,q_new,r,d,K,FISTA_options); % gradient checked.
        
        f_v = func(B_new,Q,q_new);
        
        F = [F,f_v];
        if iter > 1 
           if abs(F(end-1) - F(end))<tol
                break
           end
        end
        
        B = B_new;
        q = q_new;
        
        
        iter = iter + 1;
    end
end

function B_new = argmin_B_FISTA(Q,B,q,r,d,K,FISTA_options)


z = reshape(B,d*r,1);
 
z_func = @(x)gradient_B(x,Q,q,d,r,K);

z_new_vect = pnopt_sparsa(z_func, nonproject(), z,FISTA_options);
 
B_new = reshape(z_new_vect, d,r);



    function [f,grad_B_vec] = gradient_B(x,Q,q,d,r,K)
        B = reshape(x,d,r);
        BtB = B'*B;
        grad_B = zeros(size(B));

        for i = 1:K 
            qi = q(:,:,i);
            Qi = Q(:,:,i);
            grad_B = grad_B + 2*((B*qi'*BtB*qi+ B*qi*BtB*qi') - (Qi'*B*qi + Qi*B*qi'));
        end
         
        grad_B_vec = reshape(grad_B, d*r,1);
        f = 0;
        for i = 1:K
          f = f + sum(sum((Q(:,:,i) - B*q(:,:,i)*B').^2));
        end
    end

    
    function op = nonproject() % this part deal with the non smooth part.
    op = tfocs_prox( @f, @(x,t)prox_f(x,t) , 'vector' ); % Allow vector stepsizes


        %estimator = cell(3,1);

        function v = f(x)
            %  function value of non-smooth part

            v = 0;
        end

        function x = prox_f(v,t)

            x = v;

        end
    end 

end





function B_new = argmin_B(Q, B, q, step,tol)

    maxIter = 1000;
    K = size(Q,3);
    iter = 0;
    while (iter< maxIter)
        grad_B = zeros(size(B));
        BtB = B'*B;
        for i = 1:K 
            qi = q(:,:,i);
            Qi = Q(:,:,i);
            
            grad_B = grad_B + 2*((B*qi'*BtB*qi+ B*qi*BtB*qi') - (Qi'*B*qi + Qi*B*qi'));
        end
        B_new = B - step * grad_B;
        iter = iter + 1;
        
        if norm(B_new(:)-B(:),2)<tol
            break;
        end
        B = B_new;
    end
     
end

function fv = func(B,Q,q)
    
   fv = 0;
   K = size(Q,3);
   for i = 1:K
      fv = fv + sum(sum((Q(:,:,i) - B*q(:,:,i)*B').^2));
   end
   
end

