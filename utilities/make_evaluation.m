function [rmse, rmse_all] = make_evaluation(Xtest,Ytest, W,Q)
% input is either cell or matrix of test data, Xtest,Ytest should be both
% cell or matrix.
 numinput = nargin; 
 K = size(W,2);
 rmse_all = zeros(K,1);
 if iscell(Xtest)
%      Y_pred = cell(K,1); 
     for i = 1:K
         Xi=Xtest{i};
         Wi = W(:,i);   
     
         if numinput ==3
             Y_predi = Xi*Wi;
         else
             Y_predi = Xi*Wi + diag(Xi*Q(:,:,i)*Xi');  
         end
%          Y_pred{i} = Y_predi;
         rmse_all(i) = sqrt(mean((Y_predi - Ytest{i}).^2));
     end
     rmse = mean(rmse_all); % average rmse for all tasks.
     
 else
    Y_pred_W = Xtest*W;
    m_test = size(Ytest,1);
    if numinput ~=3
        Y_pred_Q = zeros(m_test,K);

        for ii = 1:K
           Y_pred_Q(:,ii) = diag(Xtest*Q(:,:,ii)*Xtest');  
        end

        Y_pred = Y_pred_Q + Y_pred_W;
    else
        Y_pred = Y_pred_W;
    end
    
    [rmse, rmse_all] = MTL_RMSE(Y_pred, Ytest);
        
 end
      





end