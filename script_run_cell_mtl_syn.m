% overall experiments. 

% clc;
clear;

addpath(genpath('./'))

timeflag = strrep(datestr(datetime),' ','');
disp(timeflag);
timeflag = strrep(timeflag,':','-');
timeflag = timeflag(1:end-3);

 
% configurePara = struct(...
%    'inDataDir', '../../../../AllData/multi_task/datas/',...
%    'resultDir', '../../../../AllData/multi_task/results/',...
%    'tunedParas', struct(...
%                 'lambdas'  ,  [10],...
%                  'mus' ,   [10]... 
%                  )...
% );


%% Load Data
dims  = 10 ;
task = 10;
samp = 500;
len_dim = length(dims);
datatype = 'sp'; 

Methods = {'RR','STIL','MTL_L','MTIL_L_S','MTIL_S_Ln', 'MTIL_S_Lc','MTIL_S_S','MTIL_L_Lc', 'MTIL_L_Ln'};
rmses = zeros(len_dim,length(Methods));


for i = 1:len_dim
 

dim = dims(i);

configurefile
 
dataDir = configurePara.inDataDir;
 
dataname = sprintf(strcat('Syn_', datatype, '_mtl_dim%d_task%d_samp%d.mat'),dim,task,samp);
realdata = strcat(dataDir, dataname);
load_data = load(realdata);

 
% for syn 
Xtest  = load_data.Xtest;
Ytest  = load_data.Ytest;
Xtrain = load_data.Xtrain;
Ytrain = load_data.Ytrain;



%% Run Methods

disp(sprintf('the %d iteration',i));
 

 
for kk = 1:length(Methods)
    f = str2func(strcat(Methods{kk},'_cell'));
    tic;
    rmses(i,kk) = f(Xtrain,Ytrain,Xtest,Ytest,dataname,timeflag,configurePara.(Methods{kk}));
    toc;
end

 
 
end
