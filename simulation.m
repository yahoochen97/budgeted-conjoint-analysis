if ~exist('SEED','var')
    % simulation settings
    SEED = 4;
    data_name = "Friedman";
    N = 50;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dgp effect
% use very large N
% N_exp = N;
% N = 10000;
% simulate_data;
% [dgp_effects,~]=gp_AMCE(dgp_dy, dgp_dy*0, data_name, train_x);
% N = N_exp;

% simulate profile data
simulate_data;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% difference-in-mean estimator with complete independent assumption
BIN=2;
diff_in_mean;
D = (size(train_x,2))/2;
% dim_mu = repmat(dim_mu',N,1)';
% dim_std = repmat(dim_std',N,1)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dgp_effects = mean(dgp_dy(:,1:D));
% [dgp_effects,~]=gp_point_est(BIN,raw_x,dgp_dy,dgp_dy.*0);
% dgp_effects = mean(dgp_dy(:,1:D));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% linear utility model / logistic regression
linear_utilities;
ratio = std(dgp_dy(:,1:(size(test_x,2)/2))) ./ std(lm_dy_mu);
lm_dy_mu = lm_dy_mu .* ratio;
lm_dy_std = lm_dy_std .* ratio;
[lm_mu,lm_std] = gp_AMCE(lm_dy_mu,lm_dy_std,data_name,train_x);

% build a gp preference learning model for grad
learn_HYP = 1;
n_gauss_hermite = 10;
gp_pref_grad;

ratio = std(dgp_dy(:,1:(size(test_x,2)/2))) ./ std(dy_mu);
dy_mu = dy_mu .* ratio;
dy_std = dy_std .* ratio;

ratio = std(dgp_dy(:,1:(size(test_x,2)/2))) ./ std(mu_GMM_avg);
mu_GMM_avg = mu_GMM_avg .* ratio;
sigma_GMM_avg = sigma_GMM_avg .* ratio;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dgp effects
[dgp_effects,~]=gp_AMCE(dgp_dy, dgp_dy*0, data_name, train_x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning point estimation effect
% [gp_point_mu,gp_point_std]=gp_point_est(BIN,raw_x,dy_mu,dy_std);
% gp_point_mu = mean(dy_mu(:,1:D));
% gp_point_std = sqrt(sum(dy_std(:,1:D).^2))./N;
[gp_point_mu,gp_point_std] = gp_AMCE(dy_mu,dy_std,data_name,train_x);
% gp_point_mu = mean(dy_mu);
% gp_point_std = sqrt(mean(dy_std.^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning GMM effect
% [gp_GMM_mu,gp_GMM_std]=gp_point_est(BIN,raw_x,mu_GMM_avg,sigma_GMM_avg);
% gp_GMM_mu = mean(mu_GMM_avg(:,1:D));
% gp_GMM_mu= sqrt(sum(sigma_GMM_avg(:,1:D).^2))./N;
[gp_GMM_mu,gp_GMM_std] = gp_AMCE(mu_GMM_avg,sigma_GMM_avg,data_name,train_x);
% gp_GMM_mu = mean(mu_GMM_avg);
% gp_GMM_std = sqrt(mean(sigma_GMM_avg.^2));
disp(mean((dgp_effects>=gp_GMM_mu-2*gp_GMM_std) & (dgp_effects<=gp_GMM_mu+2*gp_GMM_std)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% averaged effects
D = numel(dgp_effects);
results = array2table(zeros(4*D,3),'VariableNames',...
    {'mean','std','effect'});
results.model = cell(4*D,1);

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


results((3*D+1):(4*D),1) = num2cell(lm_mu)';
results((3*D+1):(4*D),2) = num2cell(lm_std)';
results((3*D+1):(4*D),4) = {'lm'};
results((3*D+1):(4*D),3) = num2cell(dgp_effects)';

HYP = data_name + "_N" + int2str(N) + "_TA" + int2str(test_anchor) + "_SEED" + int2str(SEED);

writetable(results,"./results/"+HYP+".csv");

% results.RMSE = abs(results.mean-results.effect);
% upper = results.mean + 1.96*results.std;
% lower = results.mean - 1.96*results.std;
% results.coverage = (lower<=results.effect) & (results.effect<=upper);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% individual effects
D = N*size(test_x,2)/2;
dgp_effects = dgp_dy(:,1:(size(test_x,2)/2));
results = array2table(zeros(4*D,3),'VariableNames',...
    {'mean','std','effect'});
results.model = cell(4*D,1);

results(1:D,1) = num2cell(reshape(repmat(dim_mu',N,1),[D,1]));
results(1:D,2) = num2cell(reshape(repmat(dim_std',N,1),[D,1]));
results(1:D,4) = {'diffinmean'};
results(1:D,3) = num2cell(reshape(dgp_effects,[D,1]));

% fig = figure(1);
% scatter(reshape(dgp_effects,[D,1]),reshape(dim_grad_mu,[D,1]));

results((1*D+1):(2*D),1) = num2cell(reshape(dy_mu,[D,1]));
results((1*D+1):(2*D),2) = num2cell(reshape(dy_std,[D,1]));
results((1*D+1):(2*D),4) = {'gppoint'};
results((1*D+1):(2*D),3) = num2cell(reshape(dgp_effects,[D,1]));

% fig = figure(2);
% scatter(reshape(dgp_effects,[D,1]),reshape(dy_mu,[D,1]));

results((2*D+1):(3*D),1) = num2cell(reshape(mu_GMM_avg,[D,1]));
results((2*D+1):(3*D),2) = num2cell(reshape(sigma_GMM_avg,[D,1]));
results((2*D+1):(3*D),4) = {'gpGMM'};
results((2*D+1):(3*D),3) = num2cell(reshape(dgp_effects,[D,1]));

results((3*D+1):(4*D),1) = num2cell(reshape(lm_dy_mu,[D,1]));
results((3*D+1):(4*D),2) = num2cell(reshape(lm_dy_std,[D,1]));
results((3*D+1):(4*D),4) = {'lm'};
results((3*D+1):(4*D),3) = num2cell(reshape(dgp_effects,[D,1]));

% fig = figure(3);
% scatter(reshape(dgp_effects,[D,1]),reshape(mu_GMM_avg,[D,1]));

writetable(results,"./results/ind_"+HYP+".csv");
