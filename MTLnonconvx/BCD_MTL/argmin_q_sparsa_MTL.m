 % use sparse to solve the optimal solution of q.
function [q_new,funcval] = argmin_q_sparsa_MTL(X,Y,W,q,B,d,K,r,FISTA_options)


z = reshape(q,r*r*K,1);

z_func = @(x)gradient_q(x,X,Y,W,B,d,K,r);

[z_new_vec, ~, output] = pnopt_sparsa(z_func, nonproject(), z, FISTA_options);

q_new = reshape(z_new_vec,r,r,K);

funcval = output.trace.f_x;


    function [f,grad_q_vec] = gradient_q(z,X,Y,W,B,d,K,r)
        
        q = reshape(z,r,r,K);
        [grad_q,f] = MTLncvBCDgradq(X,Y,W,q,B,d,K,r);
        grad_q_vec = reshape(grad_q,r*r*K,1);

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



