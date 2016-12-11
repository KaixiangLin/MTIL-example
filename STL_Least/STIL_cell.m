function min_rmse = STIL_cell(Xtrain,Ytrain, Xtest,Ytest,dataname,timeflag,configurePara)
% INPUT: Xtrain/Xtest, Ytrain/Yteste are K*1 cell. samples cells. and
% response cells/matrix
% Same function as stl_l2WsparseQ_run_cell
 

%% Set parameters
method = 'STIL';
% lambdas = [1e+8];
lambdas = configurePara.lambdas;
num_lambda = length(lambdas);
% mus  = [1e+10];
% mus  = [1000:1000:9000];
mus = configurePara.mus;
%  mus = [ 1e+9,1e+10];
 
num_mu = length(mus);


FISTA_options = pnopt_optimset(...
'debug'         , 0      ,... % debug mode 
'desc_param'    , 0.0001 ,... % sufficient descent parameter
'display'       , -500    ,... % display frequency (<= 0 for no display) 
'backtrack_mem' , 2      ,... % number of previous function values to save
'max_fun_evals' , 50000  ,... % max number of function evaluations
'max_iter'      , 100   ,... % max number of iterations
'ftol'          , 1e-6   ,... % stopping tolerance on objective function 
'optim_tol'     , 1e-6   ,... % stopping tolerance on opt
'xtol'          , 1e-9    ... % stopping tolerance on solution
);


% m = 10;
% d = 5;
% X = rand(m,d);
% Y = rand(m,1);

%% Initialization 
d = size(Xtrain{1}, 2);
K = size(Ytrain,1);

w_ini = rand(d,1);
Q_ini = rand(d,d);
 

for j = 1:d
    for i = 1:d
        if (i~=j)
          Q_ini(i,j,:) = Q_ini(j,i,:); 
        end
    end
end

% W = zeros(d,K);
% Q = zeros(d,d,K);
% AUC = zeros(num_mu, num_lambda);
% auc_tasks = zeros(num_mu, num_lambda,K);
RMSE = zeros(num_mu, num_lambda);
rmse_tasks = zeros(num_mu, num_lambda,K);
%% run sparsa 
for jj = 1:num_mu
    for ii = 1:num_lambda
 fprintf('-----------Single Task Learning sparse interaction ---lambda: %1f --= mu:%1f---------\n', lambdas(ii),mus(jj)); 
         rho1 = lambdas(ii);
         rho2 = mus(jj);    
      
         for i = 1:K
%          parfor i = 1:K

            Xt = Xtrain{i};
            Yt = Ytrain{i};
            
            [wt, Qt] = stl_l2WsparseQ(Xt,Yt,w_ini,Q_ini, rho1,rho2, FISTA_options);

            Q(:,:,i) = Qt;
            W(:,i)   = wt;
         end
%          [W, Q] = parr_interface_stl_l2WsparseQ(Xtrain, Ytrain, w_ini,Q_ini, rho1,rho2,d,K,FISTA_options);
        [RMSE(jj,ii), rmse_tasks(jj,ii,:)] = make_evaluation(Xtest,Ytest, W,Q);
%          [AUC(jj,ii), auc_tasks(jj,ii,:)] = make_evaluation_AUC(Xtest,Ytest,1, W,Q);
         W_all{jj,ii} = W;
         Q_all{jj,ii} = Q;
    end
end
 

 
 
min_rmse = min(min(RMSE));
disp(sprintf('The min RMSE of %s is %1.5f  ',method,min_rmse  ));
 

end 