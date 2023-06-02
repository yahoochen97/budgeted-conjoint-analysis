if ~exist('SEED','var')
    % simulation settings
    SEED = 1;
    data_name = "2Dplane";
    N = 1000;
end

% add gpml tool box; clear env;
close all;
% add gpml path
addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
addpath("utilities");
FONTSIZE=16;

rng(SEED);

% simulate profile data
simulate_data;

% split train/test data
train_x = transformed_x;
train_y = pair_y;
test_x = train_x;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% difference-in-mean estimator with complete independent assumption
BIN=10;
diff_in_mean;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dgp effect
BIN=10;
[dgp_effects,~]=gp_point_est(BIN,raw_x,dgp_dy,dgp_dy.*0);

fig = figure(1);
scatter(dgp_effects,dim_mu);

% build a gp preference learning model for grad
learn_HYP = 1;
gp_pref_grad;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning point estimation effect
BIN=10;
[gp_point_mu,gp_point_std]=gp_point_est(BIN,raw_x,dy_mu,dy_std);

fig = figure(2);
scatter(dgp_effects,gp_point_mu);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning GMM effect
BIN=10;
[gp_GMM_mu,gp_GMM_std]=gp_point_est(BIN,raw_x,mu_GMM_avg,sigma_GMM_avg);

fig = figure(3);
scatter(dgp_effects,gp_GMM_mu);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

HYP = data_name + "_N" + int2str(N) + "_d" + int2str(d) + "_SEED" + int2str(SEED);

writetable(results,"./results/"+HYP+".csv");
