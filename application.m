if ~exist('data_name','var')
    data_name = "hainmueller_candidate";
end

maxNumCompThreads(1);

% add gpml tool box; clear env;
close all;
% add gpml path
addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
addpath("utilities");
FONTSIZE=16;
rng(12345);

file_name = "./data/" + data_name + ".csv";
opts = detectImportOptions(file_name);
data = readtable(file_name, opts);

train_x = data{:,1:(end-1)};
train_y = 2*data{:,end}-1;
test_x = train_x;

% build a gp preference learning model for grad
learn_HYP = 1;
n_gauss_hermite = 10;
gp_pref_grad;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning GMM effect
[gp_GMM_mu,gp_GMM_std]=gp_point_est(BIN,raw_x,mu_GMM_avg,sigma_GMM_avg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = size(train_x,2)/2;
results = array2table(zeros(D,2),'VariableNames',...
    {'mean','std'});
results.model = cell(D,1);

results((0*D+1):(1*D),1) = num2cell(gp_GMM_mu)';
results((0*D+1):(1*D),2) = num2cell(gp_GMM_std)';
results((0*D+1):(1*D),3) = {'gpGMM'};

writetable(results,"./results_application/"+data_name+".csv");
