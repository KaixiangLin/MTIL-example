% use sparse to solve the optimal solution of q.
function [W_new,funcval] = argmin_B_sparsa_MTL(X,Y,W,q,B,d,K,r,FISTA_options)


z = reshape(B,d*r,1);

z_func = @(x)gradient_B(x,X,Y,W,q,d,K,r);

[z_new_vec, ~,output] = pnopt_sparsa(z_func, nonproject(), z, FISTA_options);

funcval = output.trace.f_x;

W_new = reshape(z_new_vec,d,r);




    function [f,grad_B_vec] = gradient_B(z,X,Y,W,q,d,K,r)
        
%         grad_B = zeros(d, K);
        B = reshape(z,d,r);
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
%                
%                XtjB  = Xtj*B; % 1*r
%                grad_B      = grad_B + (Xtj'*ZZ)*(XtjB*qt') + (Xtj'*ZZ')*(XtjB*qt)...
%                             + (Xtj')*((XtjB*qt')*XtjB')*(XtjB*qt)...
%                             + (Xtj')*((XtjB*qt)* XtjB')*(XtjB*qt');
%                         
%                f = f + 1/2*Xtemp*Xtemp;
%             end
%         end
        [grad_B,f] = MTLncvBCDgradB(X,Y,W,q,B,d,K,r);
        grad_B_vec = reshape(grad_B,d*r,1);

    end

    function op = nonproject() % this part deal with the non smooth part.
    op = tfocs_prox( @f, @(x,t)prox_f(x,t) , 'vector' ); % Allow vector stepsizes
 

        function v = f(x)
            %  function value of non-smooth part

            v = 0;
        end

        function x = prox_f(v,t)

            x = v;

        end
    end 
 
end



