% use sparse to solve the optimal solution of q.
function [W_new,funcval] = argmin_W_sparsa_MTL_cell(X,Y,W,q,B,d,K,r,rho1,FISTA_options)


z = reshape(W,d*K,1);

z_func = @(x)gradient_W(x,X,Y,B,q,K,d,r );

[z_new_vec,~,output] = pnopt_sparsa(z_func, project(rho1, d, K), z, FISTA_options);

W_new = reshape(z_new_vec,d,K);

funcval = output.trace.f_x;


    function [f,grad_W_vec] = gradient_W(z,X,Y,B,q,K,d,r)
        
%         grad_W = zeros(d, K);
        W = reshape(z,d,K);
%         f = 0;
%         for i = 1:K
%             Xt = X{i};
%             mt = size(Xt,1);
%             Wt = W(:,i);
%             qt = q(:,:,i); % r*r
%             Qt = B*qt*B';
%             Yt = Y{i};
% 
%             for j = 1:mt
%                Xtj = Xt(j,:);  % 1 * d
%                ZZ = Xtj*Wt - Yt(j);
%                Xtemp = (Xtj*Qt*Xtj' + ZZ);
%                grad_W(:,i) = grad_W(:,i) + Xtj'*Xtemp;
% 
%                f = f + 1/2*Xtemp*Xtemp;
%             end
%         end
        [grad_W,f] = ncvBCDgradW(X,Y,W,q,B,d,K,r);
        grad_W_vec = reshape(grad_W,d*K,1);

    end

    function op = project(rho1, d, K) % this part deal with the non smooth part.
    op = tfocs_prox( @f, @(x,t)prox_f(rho1,x,t) , 'vector' ); % Allow vector stepsizes


        %estimator = cell(3,1);

        function v = f(x)
            %  function value of non-smooth part


%             W = x(1:d*K,1); 
            f_W = norm(x,1) * rho1;
            v   = f_W;    
        end

        function x = prox_f(rho1,v,t)

            % Recover weight matrix w and interacting tensor Q.

    

            % this projection calculates
            % argmin_z = 1/2\|z-v\|_2^2 + beta \|z\|_1
            % z: solution
            % l1_comp_val: value of l1 component (\|z\|_1)
            x = sign(v).*max(0,abs(v)- rho1*t);

 

        end
    end

 
end



