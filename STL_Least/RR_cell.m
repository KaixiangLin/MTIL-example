%% This script is tune the parameters for single task learning by least square method.

function min_rmse = RR_cell(Xtrain,Ytrain,Xtest,Ytest,dataname,timeflag,configurePara)
% clc;               
% clear;
% Same function as STL_Least_run_cell
% addpath(genpath('../../'))

 

%% Set parameters
% lambdas = [0.1, 1, 10, 100, 1000, 1e+4,1e+5,1e+6,1e+7,1e+8];
% lambdas = [0.1];
lambdas = configurePara.lambdas;
num_lambda = length(lambdas);
method = 'RR';



%% Initialization
d = size(Xtrain{1}, 2);
K = size(Ytrain,1);

% AUC = zeros(num_lambda,1);
RMSE = zeros(num_lambda,1);
% auc_tasks = zeros(num_lambda,K);

rng(0);
W_ini = rand(d,K);

%save data
W_all = cell(num_lambda,1);
 
%% Grid search
% parfor i = 1:num_lambda
for i = 1:num_lambda
    lambda = lambdas(i);
    
    fprintf('-----------Single Task Learning Least Square --- lambda: %1f -----------\n', lambda);
    
    
    % Learn model
    W = least_method_cell(Xtrain, Ytrain, W_ini, lambda);
    
%     [RMSE(i), rmse_tasks(i,:)] = MTL_RMSE(Y_pred, Ytest);
    [RMSE(i), rmse_tasks(i,:)] = make_evaluation(Xtest,Ytest, W);
%     [AUC(i), auc_tasks(i,:)] = make_evaluation_AUC(Xtest,Ytest,1,W);
    W_all{i} = W;

end

%% Save Data
 
min_rmse = min(RMSE);

 
disp(sprintf('The min RMSE of %s is %1.5f  ',method,min_rmse  ));
 

