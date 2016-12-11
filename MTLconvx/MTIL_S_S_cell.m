% Tune parameters for Multi task learning for Convex formulation

function [min_rmse,output] = MTIL_S_S_cell(Xtrain,Ytrain,Xtest,Ytest,dataname,timeflag, configurePara)
% clc;
% clear;
tic;
% addpath(genpath('../../'))

 

%% Set parameters
 
% lambdas = [0.1, 1, 10, 100, 1000, 1e+4,1e+5,1e+6,1e+7 ,1e+8,1e+9,1e+10];
% num_lambda = length(lambdas);
% mus = [10,1e+2,1e+3,1e+4,1e+5,1e+6,1e+7,1e+8,1e+9,1e+10];
% num_mu = length(mus);

% tunedParas = struct(...
% 'lambdas'  ,  [1,10,1e+2,1e+3,1e+4],...
% 'mus' ,       [1,10,1e+2,1e+3,1e+4]... 
% );
tunedParas = configurePara.tunedParas;

method = 'MTIL_S_S';

FISTA_options = pnopt_optimset(...
'debug'         , 0      ,... % debug mode 
'desc_param'    , 0.0001 ,... % sufficient descent parameter
'display'       , -300    ,... % display frequency (<= 0 for no display) 
'backtrack_mem' , 2      ,... % number of previous function values to save
'max_fun_evals' , 50000  ,... % max number of function evaluations
'max_iter'      , 100   ,... % max number of iterations
'ftol'          , 1e-6   ,... % stopping tolerance on objective function 
'optim_tol'     , 1e-6   ,... % stopping tolerance on opt
'xtol'          , 1e-9    ... % stopping tolerance on solution
);

%% Initialization
d = size(Xtrain{1}, 2);
K = size(Ytrain,1);
 
 
rng(0);
W_ini = rand(d,K);
Q_ini = rand(d,d,K);

if sum(Q_ini(:)~=0)
% symetric projection
for j = 1:d
    for i = 1:d
        if (i~=j)
          Q_ini(i,j,:) = Q_ini(j,i,:); 
        end
    end
end
end

% parallel computing parameters  %%%%%%%%%%%%%%%%%
lenPara = 1;
paraNames = fieldnames(tunedParas);
num_paras = length(paraNames);
len_paras = zeros(num_paras,1);
dividend   = ones(num_paras,1);   
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
% parallel computing parameters  %%%%%%%%%%%%%%%%%

 
% Save models
W_all = cell(lenPara,1);
Q_all = cell(lenPara,1);
f_value = cell(lenPara,1);
RMSE  = cell(lenPara,1);
rmse_tasks = cell(lenPara,1);
%% Grid search
 
% parfor  i = 1:lenPara
for  i = 1:lenPara

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
        
        lambda = parameters.(paraNames{1});
        mu     = parameters.(paraNames{2});
%         fprintf('-----------MTIL_S_S --- lambda: %1.2f -- mu:%1.2f ---\n', lambda,mu);

        
        % Learn model
        [W,Q,f] = argmin_l21WGLQ_Sparsa_mex_MTLExp_cell(Xtrain, Ytrain, W_ini,Q_ini,lambda, mu, FISTA_options);
 

        % Calcuate multi task Average RMSE
        [RMSE{i}, rmse_tasks{i}] = make_evaluation(Xtest,Ytest, W,Q);
        
        W_all{i} = W;
        Q_all{i} = Q;
        f_value{i} = f;

 
end

%% Save Data

 
[min_rmse,index] = min(cell2mat(RMSE));
output.Q = Q_all{index};
output.W = W_all{index};
output.f = f_value{index};

elapsedTime = toc;
disp(sprintf('The min RMSE of %s is %1.5f  ',method,min_rmse));
 
 
end
