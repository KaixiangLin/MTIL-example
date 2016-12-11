% use sparse to solve the optimal solution of q.
function [W_new,funcval] = argmin_Breg_sparsa_MTL_cell(X,Y,W,q,B,d,K,r,lambdaB,FISTA_options)


z = reshape(B,d*r,1);

z_func = @(x)gradient_B(x,X,Y,W,q,d,K,r,lambdaB);

[z_new_vec, ~,output] = pnopt_sparsa(z_func, nonproject(), z, FISTA_options);

funcval = output.trace.f_x;

W_new = reshape(z_new_vec,d,r);




    function [f,grad_B_vec] = gradient_B(z,X,Y,W,q,d,K,r,lambdaB)
        B = reshape(z,d,r);

        [grad_B,f] = ncvBCDgradBreg(X,Y,W,q,B,d,K,r,lambdaB);
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



