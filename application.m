if ~exist('data_name','var')
    data_name = "hainmueller_immigrant"; % "hainmueller_candidate";
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

raw_x = data{:,1:(end-1)};
d = size(raw_x,2)/2;
train_x = transformdummy(raw_x);
train_y = 2*data{:,end}-1;
test_x = train_x;

diff_in_mean;

% build a gp preference learning model for grad
learn_HYP = 1;
n_gauss_hermite = 10;
gp_pref_grad;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning GMM effect
[gp_GMM_mu,gp_GMM_std]=gp_AMCE(mu_GMM_avg,sigma_GMM_avg, data_name, train_x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = size(train_x,2)/2;
results = array2table(zeros(2*D,2),'VariableNames',...
    {'mean','std'});
results.model = cell(2*D,1);

results(1:D,1) = num2cell(dim_mu)';
results(1:D,2) = num2cell(dim_std)';
results(1:D,3) = {'diffinmean'};

results((1*D+1):(2*D),1) = num2cell(gp_GMM_mu)';
results((1*D+1):(2*D),2) = num2cell(gp_GMM_std)';
results((1*D+1):(2*D),3) = {'gpGMM'};

writetable(results,"./results_application/"+data_name+".csv");
