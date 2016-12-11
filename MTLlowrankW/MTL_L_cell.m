% Tune parameters for Multi task learning for low rank regularization on
% W. 

function min_rmse = MTL_L_cell(Xtrain,Ytrain,Xtest,Ytest,dataname,timeflag,configurePara)
% clc;
% clear;
 
%% Set parameters
% lambdas = [0.1, 0.5, 1, 2, 3, 4, 5];
% lambdas =  [0.1, 1, 10, 100, 1000, 1e+4,1e+5,1e+6,1e+7,1e+8,1e+9,1e+10];
lambdas = configurePara.lambdas;
% lambdas =  [0.1];
num_lambda = length(lambdas);
method = 'MTL_L';

FISTA_options = pnopt_optimset(...
'debug'         , 0      ,... % debug mode 
'desc_param'    , 0.0001 ,... % sufficient descent parameter
'display'       , -10    ,... % display frequency (<= 0 for no display) 
'backtrack_mem' , 10     ,... % number of previous function values to save
'max_fun_evals' , 50000  ,... % max number of function evaluations
'max_iter'      , 1000   ,... % max number of iterations
'ftol'          , 1e-3   ,... % stopping tolerance on objective function 
'optim_tol'     , 1e-6   ,... % stopping tolerance on opt
'xtol'          , 1e-9    ... % stopping tolerance on solution
);

%% Initialization
d = size(Xtrain{1}, 2);
K = size(Ytrain,1);

% AUC = zeros(num_lambda,1);
RMSE = zeros(num_lambda,1);
rng(0);
W_ini = rand(d,K);
%save data
W_all = cell(num_lambda,1);
% auc_tasks = zeros(num_lambda,K);
rmse_tasks = zeros(num_lambda,K);
f_value = cell(num_lambda,1);
%% Grid search
% parfor i = 1:num_lambda
for i = 1:num_lambda

    lambda = lambdas(i);
    
    fprintf('-----------Multi Task Learning Low rank W --- lambda: %1.2f -----------\n', lambda);
    
    
    % Learn model
    [W,f] = argmin_lowrank_W_exp_cell(Xtrain, Ytrain, W_ini, lambda, FISTA_options);

 
 
    [RMSE(i), rmse_tasks(i,:)] = make_evaluation(Xtest,Ytest, W);
    W_all{i} = W;
    f_value{i} = f;
end


min_rmse = min(RMSE);
disp(sprintf('The min RMSE of %s is %1.5f  ',method,min_rmse  ));
 

end
