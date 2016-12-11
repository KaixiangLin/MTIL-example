% Tune parameters for Multi task learning for Convex formulation

function [min_rmse,output] = MTIL_S_Ln_cell(Xtrain,Ytrain,Xtest,Ytest,dataname,timeflag,configurePara)
% clc;
% clear;
% INPUT: Xtrain K*1 cell, Ytrain K*1 cell. 

% addpath(genpath('../../'))
% addpath(genpath('./'))
 

%% Set parameters
 

d = size(Xtrain{1}, 2);
K = size(Ytrain,1);


 


tunedParas = configurePara.tunedParas;

rankindex = 4; % used when initialize the B and q.


 
method = 'MTIL_S_Ln';
FISTA_OPT = pnopt_optimset(...
        'debug'         , 0      ,... % debug mode 
        'desc_param'    , 0.0001 ,... % sufficient descent parameter
        'display'       , -10    ,... % display frequency (<= 0 for no display) 
        'backtrack_mem' , 10     ,... % number of previous function values to save
        'max_fun_evals' , 50000  ,... % max number of function evaluations
        'max_iter'      , 10   ,... % max number of iterations
        'ftol'          , 1e-3   ,... % stopping tolerance on objective function 
        'optim_tol'     , 1e-6   ,... % stopping tolerance on opt
        'xtol'          , 1e-9    ... % stopping tolerance on solution
        );
maxIter = 100;
maintol = 1e-3;

%% Initialization
 
 
 

 

% test
rng(0) 
W_ini = rand(d,K);
Q_ini = rand(d,d,K);
% [B_ini,q_ini,F] = block_coordinate(Q, parameters.rank,FISTA_options)

%%%%%%%%%
% % using convex results to initialization
% timeflag1 = '27-Mar-201611-24';  % test
% dataload = load(sprintf(strcat(configurePara.resultDir,'%s_%s_%s'), 'MTIL_L_Lc', timeflag1 ,dataname));
% cvx_rmse = cell2mat(dataload.RMSE);
% cvx_idx  = find(cvx_rmse==min(min(cvx_rmse)));
% cvx_W_all    = dataload.W_all;
% cvx_Q_all    = dataload.Q_all;
% W_ini = cvx_W_all{cvx_idx};
% Q_ini = cvx_Q_all{cvx_idx};

%%%%%%%%%%%%%
lenPara = 1;
paraNames = fieldnames(tunedParas);
num_paras = length(paraNames);
len_paras = zeros(num_paras,1);
dividend   = ones(num_paras,1);  % the 

para_i = cell(num_paras,1);
for i = 1:num_paras
    para_i{i} = getfield(tunedParas, paraNames{i});
    lenPara = lenPara*length(para_i{i});
    len_paras(i) = length(para_i{i});  % lenght of each parameter array
    
    if i ~= num_paras
        for jj = i+1:num_paras
            dividend(i) = dividend(i) * length(getfield(tunedParas, paraNames{jj}));
        end
    else
        dividend(i) = len_paras(i);
    end
end
% Save models
W_all = cell(lenPara,1);
Q_all = cell(lenPara,1);
B_all = cell(lenPara,1);
q_all = cell(lenPara,1);
f_value = cell(lenPara,1);
RMSE  = cell(lenPara,1);
rmse_tasks = cell(lenPara,1);

%% Grid search
% parfor i = 1:lenPara
for i = 1:lenPara

        paras_indexs = zeros(num_paras,1); % the real index in each array
        for j = 1:num_paras
            if j == num_paras
                paras_indexs(j) = mod(i,len_paras(j));
                  
            else
                temp = ceil(i/dividend(j));
                paras_indexs(j) = mod(temp,len_paras(j));
            end 
            if paras_indexs(j)== 0
                    paras_indexs(j) = len_paras(j);
            end
        end
        

        parameters = struct();
        for jj = 1:num_paras
            para_array = para_i{jj}; % current parameters arrays.
            parameters.(paraNames{jj}) = para_array(paras_indexs(jj)) ;
        end
    
        parameters.FISTA_options = FISTA_OPT;
        
        parameters.maxIter = maxIter;
        parameters.maintol = maintol;

         
        [B_ini,q_ini,F] = block_coordinate(Q_ini, parameters.(paraNames{rankindex}),FISTA_OPT);

 fprintf('----%d iter--MTIL_S_Ln   -lambdaW:%1.2f - lambdaB:%1.2f-lambdaq:%1.2f - ranks: %1.2f ---\n',...
     i, paras_indexs(1),paras_indexs(2),paras_indexs(3),paras_indexs(4));
%         % Initialization % test
%         B_ini = zeros(d,parameters.rank);
%         q_ini = zeros(parameters.rank,parameters.rank,K);

        
        % Learn model
        [W,Q,B,q,funcval] = l21W_LRnQreg_block_main_MTL_cell(Xtrain, Ytrain, W_ini,B_ini,q_ini, parameters);
        
        % Calcuate multi task RMSE.  
        [RMSE{i}, rmse_tasks{i}] = make_evaluation(Xtest,Ytest, W,Q);
        f_value{i} = funcval;
        W_all{i} = W;
        Q_all{i} = Q;
        B_all{i} = B;
        q_all{i} = q;
 
end


%% Save Data
% min_rmse = min(cell2mat(RMSE));
[min_rmse,index] = min(cell2mat(RMSE));
output.Q = Q_all{index};
output.W = W_all{index};
output.B = B_all{index};
output.q = q_all{index};
output.f = f_value{index};

disp(sprintf('The min RMSE of %s is %1.5f  ',method,min_rmse  ));

% save(sprintf(strcat(configurePara.resultDir,'%s_%s_%s'), method, timeflag,dataname));
% ,'maxIter','FISTA_OPT','maintol','f_value','rmse_tasks', 'W_all','Q_all','B_all','q_all',...'RMSE','tunedParas','min_rmse' 
end
