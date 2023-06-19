if ~exist('SEED','var')
    % simulation settings
    SEED = 1;
    data_name = "twoDplane";
    N = 100;
    test_anchor = 0;
end

maxNumCompThreads(1);

% add gpml tool box; clear env;
close all;
% add gpml path
addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
addpath("utilities");
FONTSIZE=16;

rng(12345+SEED);

% simulate profile data
simulate_data;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% difference-in-mean estimator with complete independent assumption
BIN=10;
diff_in_mean;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dgp effect
[dgp_effects,~]=gp_point_est(BIN,raw_x,dgp_dy,dgp_dy.*0);

% build a gp preference learning model for grad
learn_HYP = 1;
n_gauss_hermite = 10;
gp_pref_grad;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning point estimation effect
[gp_point_mu,gp_point_std]=gp_point_est(BIN,test_x,dy_mu,dy_std);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning GMM effect
[gp_GMM_mu,gp_GMM_std]=gp_point_est(BIN,test_x,mu_GMM_avg,sigma_GMM_avg);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = numel(dim_mu);
results = array2table(zeros(3*D,3),'VariableNames',...
    {'mean','std','effect'});
results.model = cell(3*D,1);

results(1:D,1) = num2cell(dim_mu)';
results(1:D,2) = num2cell(dim_std)';
results(1:D,4) = {'diffinmean'};
results(1:D,3) = num2cell(dgp_effects)';

results((1*D+1):(2*D),1) = num2cell(gp_point_mu)';
results((1*D+1):(2*D),2) = num2cell(gp_point_std)';
results((1*D+1):(2*D),4) = {'gppoint'};
results((1*D+1):(2*D),3) = num2cell(dgp_effects)';

results((2*D+1):(3*D),1) = num2cell(gp_GMM_mu)';
results((2*D+1):(3*D),2) = num2cell(gp_GMM_std)';
results((2*D+1):(3*D),4) = {'gpGMM'};
results((2*D+1):(3*D),3) = num2cell(dgp_effects)';

% results.RMSE = abs(results.mean-results.effect);
% upper = results.mean + 1.96*results.std;
% lower = results.mean - 1.96*results.std;
% results.coverage = (lower<=results.effect) & (results.effect<=upper);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% D = N*size(test_x,2)/2;
% dgp_effects = dgp_dy(:,1:(size(test_x,2)/2));
% results = array2table(zeros(3*D,3),'VariableNames',...
%     {'mean','std','effect'});
% results.model = cell(3*D,1);
% 
% results(1:D,1) = num2cell(reshape(dim_grad_mu,[D,1]));
% results(1:D,2) = num2cell(reshape(dim_grad_std,[D,1]));
% results(1:D,4) = {'diffinmean'};
% results(1:D,3) = num2cell(reshape(dgp_effects,[D,1]));
% 
% fig = figure(1);
% scatter(reshape(dgp_effects,[D,1]),reshape(dim_grad_mu,[D,1]));
% 
% results((1*D+1):(2*D),1) = num2cell(reshape(dy_mu,[D,1]));
% results((1*D+1):(2*D),2) = num2cell(reshape(dy_std,[D,1]));
% results((1*D+1):(2*D),4) = {'gppoint'};
% results((1*D+1):(2*D),3) = num2cell(reshape(dgp_effects,[D,1]));
% 
% fig = figure(2);
% scatter(reshape(dgp_effects,[D,1]),reshape(dy_mu,[D,1]));
% 
% results((2*D+1):(3*D),1) = num2cell(reshape(mu_GMM_avg,[D,1]));
% results((2*D+1):(3*D),2) = num2cell(reshape(sigma_GMM_avg,[D,1]));
% results((2*D+1):(3*D),4) = {'gpGMM'};
% results((2*D+1):(3*D),3) = num2cell(reshape(dgp_effects,[D,1]));
% 
% fig = figure(3);
% scatter(reshape(dgp_effects,[D,1]),reshape(mu_GMM_avg,[D,1]));

HYP = data_name + "_N" + int2str(N) + "_TA" + int2str(test_anchor) + "_SEED" + int2str(SEED);

writetable(results,"./results/"+HYP+".csv");
