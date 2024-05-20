if ~exist('SEED','var')
    % simulation settings
    SEED = 15;
    data_name = "hainmueller_immigrant"; %  "hainmueller_candidate"; %
    policy_name = "UNIFORM";
    TOTAL_SIZE = 250;
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
BATCH_SIZE = 1; % acquire 1 new data per iteration
SAVE_BATCH = 25; % save results every 25 iterations

rng(SEED+42);

file_name = "./data/" + data_name + ".csv";
opts = detectImportOptions(file_name);
data = readtable(file_name, opts);

raw_x = data{:,1:(end-1)};
d = size(raw_x,2)/2;
train_x = transformdummy(raw_x);
train_y = 2*data{:,end}-1;
test_x = train_x;
N = size(train_x,1);
x_pop = train_x;
y_pop = train_y;

% build model with initial batch
INIT_SIZE = 200;
idx_selected = (N+1-INIT_SIZE:N);
train_x = x_pop(idx_selected,:);
train_y = y_pop(idx_selected,:);
idx_other = setdiff(1:N, idx_selected);
test_x = x_pop(idx_other,:);
test_y = y_pop(idx_other,:);
learn_HYP = 1;
n_gauss_hermite = 10;
gp_pref_grad;
N = N - INIT_SIZE;
x_pop = test_x;
y_pop = test_y;

% initial batch is complete randomization
INIT_SIZE = 25;
idx_selected = [];
idx_init = policy_uniform(1:N, INIT_SIZE);
idx_selected = [idx_selected, idx_init];
train_x = x_pop(idx_selected,:);
train_y = y_pop(idx_selected,:);
idx_other = setdiff(1:N, idx_selected);
test_x = x_pop(idx_other,:);
test_y = y_pop(idx_other,:);

% true dgp effect with whole population
file_name = "./results_application/"+data_name+".csv";
opts = detectImportOptions(file_name);
data = readtable(file_name, opts);
dgp_effects = data.mean(strcmp(data.model, "gpGMM"));

% adaptively acquire new data as batches
ITERATIONS = (TOTAL_SIZE-INIT_SIZE)/BATCH_SIZE;
epsilon = 0.05;
ACC = [];

for iter=1:ITERATIONS 
   % policy for data acquisition
   disp("search iter " + iter);
   
   % current gp model but fix hyp
   learn_HYP = 0;
   [ymu,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, ...
                covfunc, likfunc, train_x, train_y, test_x);
   [mu_GMM_avg,sigma_GMM_avg, mu_GMM,sigma_GMM,...
       dy_mu, dy_std, df_mu, df_K, ks, ws] = g_GMM(n_gauss_hermite, ...
       hyp,inffunc,meanfunc,covfunc, likfunc, train_x, train_y, test_x);

   if strcmp(policy_name, "UNIFORM")
       % randomization policy
       idx_cur = policy_uniform(idx_other, BATCH_SIZE);
   elseif strcmp(policy_name, "BALD")
       % Bayesian active learning by disagreement
       ps = normcdf(fmu./sqrt(1+fs2));
       C = sqrt(pi*log(2)/2);
       IG_p = -ps.*log2(ps) - (1-ps).*log2(1-ps) - ...
            C./sqrt(C^2+fs2).*exp(-fmu.^2./(C^2+fs2)/2);
       idx_cur = epsilon_greedy(IG_p, BATCH_SIZE, epsilon);
       idx_cur = idx_other(idx_cur);
   elseif strcmp(policy_name, "DE")
       % maximize diffential entropy for latent utility
       U_f = arrayfun(@(i)det(squeeze(df_K(i,:,:))),1:size(test_x,1));
       DE_f = log(sqrt(U_f)); % ln(sigma) + ln(2 pi) / 2 + 0.5;
       idx_cur = epsilon_greedy(DE_f, BATCH_SIZE, epsilon);
       idx_cur = idx_other(idx_cur);
   elseif strcmp(policy_name, "GRADDE")
       % maximize diffential entropy for marginal effect
       DE_g = sum(log(sigma_GMM_avg),2);
       idx_cur = epsilon_greedy(DE_g, BATCH_SIZE, epsilon);
       idx_cur = idx_other(idx_cur);
    elseif strcmp(policy_name, "UCB")
       % maximize upper confidence bound
       UCB = abs(fmu+1.96*sqrt(fs2)); 
       idx_cur = epsilon_greedy(UCB, BATCH_SIZE, epsilon);
       idx_cur = idx_other(idx_cur);
   end
   
   % append new acquisition to dataset
   idx_selected = [idx_selected, idx_cur];
   train_x = x_pop(idx_selected,:);
   train_y = y_pop(idx_selected,:);
   idx_other = setdiff(1:N, idx_selected);
   test_x = x_pop(idx_other,:);
   test_y = y_pop(idx_other,:);
   
   % save results every 25 samples   
   if mod(numel(idx_selected), SAVE_BATCH)==0
        [ymu,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, ...
                covfunc, likfunc, train_x, train_y, test_x);
        ACC = [ACC, mean((ymu>=0)==(test_y==1))];
   end
end

HYP = data_name + "_" + policy_name + "_SEED" + int2str(SEED);
csvwrite("./results_application/ACC_"+HYP+".csv", ACC');

function idx_cur = epsilon_greedy(IG, BATCH_SIZE, epsilon)
    r = rand;
    if r>=epsilon
        [~,idx_cur]=maxk(IG,BATCH_SIZE);
    else
       idx_cur = policy_uniform(1:size(IG), BATCH_SIZE); 
    end
end
