if ~exist('SEED','var')
    % simulation settings
    SEED = 15;
    data_name = "Friedman";
    policy_name = "GRADBALD";
    N = 1000;
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

% generate data pool with complete randomization
simulate_data;
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
dgp_dy = dgp_dy(idx_other,:);
dgp_f = dgp_f(idx_other,:);
dgp_df = dgp_df(idx_other,:);
dgp_p = dgp_p(idx_other,:);
raw_x = raw_x(idx_other,:);
pair_y = pair_y(idx_other,:);
transformed_x = transformed_x(idx_other,:);

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
D = size(train_x,2)/2;
% [dgp_effects,~]=gp_point_est(BIN,raw_x,dgp_dy,dgp_dy.*0);
[dgp_effects,~]=gp_AMCE(dgp_dy(idx_other,:),dgp_dy(idx_other,:)*0,data_name, x_pop(idx_other,:));

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
   if mod(numel(idx_selected), SAVE_BATCH)==0
      ACC = [ACC, mean((ymu>=0)==(test_y==1))];
   end
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
       % [~,idx_cur]=maxk(IG_p,BATCH_SIZE);
       % idx_cur = softmax(IG_p, BATCH_SIZE);
       idx_cur = epsilon_greedy(IG_p, BATCH_SIZE, epsilon);
       idx_cur = idx_other(idx_cur);
%    elseif strcmp(policy_name, "US")
%        % maximize uncertainty for latent utility
%        U_f = arrayfun(@(i)det(squeeze(df_K(i,:,:))),1:size(test_x,1));
%        idx_cur = epsilon_greedy(U_f, BATCH_SIZE, epsilon);
%        idx_cur = idx_other(idx_cur);
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
%    elseif strcmp(policy_name, "GRADUS")
%        % maximize uncertainty for marginal effect
%        U_g = sum(sigma_GMM_avg,2);
%        idx_cur = epsilon_greedy(U_g, BATCH_SIZE, epsilon);
%        idx_cur = idx_other(idx_cur);
   elseif strcmp(policy_name, "GRADBALD")
       % information gain of marginal effects
       ps = normcdf(fmu./sqrt(1+fs2));
       IG_g = -ps.*log2(ps) - (1-ps).*log2(1-ps);
       
       for k=1:size(test_x,1)
            % compute new GMMs
            xs = test_x(k,:);
            [mu_GMM1,sigma_GMM1, ~, ~, ~, ~,~,~] = g_GMM(n_gauss_hermite,hyp,...
                    inffunc,meanfunc,covfunc, likfunc, [train_x; xs], [train_y;1], xs);
            [mu_GMM0,sigma_GMM0, ~, ~, ~, ~,~,~] = g_GMM(n_gauss_hermite,hyp,...
                    inffunc,meanfunc,covfunc, likfunc, [train_x; xs], [train_y;-1], xs);
%             mu_GMM1 = squeeze(mu_GMM1); mu_GMM0 = squeeze(mu_GMM0);
%             sigma_GMM1 = squeeze(sigma_GMM1); sigma_GMM0 = squeeze(sigma_GMM0);
            for i=1:n_gauss_hermite
                for j=1:n_gauss_hermite
                    g_bar = sqrt(2)*sigma_GMM(k,:,i)*ks(j) + mu_GMM(k,:,i);
%                     p_1 = 0; p_0 = 0;
                    % compute p(g|y,x,D)
%                     for it=1:n_gauss_hermite
%                         for l=1:D
%                             p_1 = p_1 + ws(it)*normpdf(g_bar(l),mu_GMM1(l,it),diag(sigma_GMM1(l,it)));
%                             p_0 = p_0 + ws(it)*normpdf(g_bar(l),mu_GMM0(l,it),diag(sigma_GMM0(l,it)));
%                         end 
%                     end
                    p_1 = mvnpdf(g_bar,mu_GMM1,diag(sigma_GMM1));
                    p_0 = mvnpdf(g_bar,mu_GMM0,diag(sigma_GMM0));
                        
                    % compute E[H[y|x,g]]  
                    p_k = 1/(1+p_0/p_1*(1-ps(k))/ps(k));
                    p_k = max(min(p_k,1-1e-12),1e-12);
                    h = -p_k*log2(p_k) - (1-p_k)*log2(1-p_k);
                    if ~isnan(h), IG_g(k) = IG_g(k) - ws(i)*ws(j)*h; end
                end
            end   
       end
       
       % [~,idx_cur]=maxk(IG_g,BATCH_SIZE);
       % idx_cur = softmax(IG_g, BATCH_SIZE);
       idx_cur = epsilon_greedy(IG_g, BATCH_SIZE, epsilon);
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
       HYP = data_name + "_N" + int2str(N) + "_S" + int2str(numel(idx_selected)) + "_" + policy_name + "_SEED" + int2str(SEED);
       
       % get current estimation
       idx_test = setdiff(1:N, idx_init);
       [mu_GMM_avg,sigma_GMM_avg, mu_GMM,sigma_GMM,...
           dy_mu, dy_std, df_mu, df_K, ks, ws] = g_GMM(n_gauss_hermite, ...
           hyp,inffunc,meanfunc,covfunc, likfunc, train_x, train_y, x_pop(idx_test,:));
       
       % report individualized effect estimation
        [gp_GMM_mu,gp_GMM_std]=gp_AMCE(mu_GMM_avg,sigma_GMM_avg,data_name, x_pop(idx_test,:));
        D = numel(dgp_effects);
        results = array2table(zeros(D,3),'VariableNames',...
            {'mean','std','effect'});
        results.policy = repmat(string(policy_name),[D 1]);

        results(:,1) = num2cell(gp_GMM_mu)';
        results(:,2) = num2cell(gp_GMM_std)';
        results(:,3) = num2cell(dgp_effects)';

        writetable(results,"./results2/"+HYP+".csv");
               
        % report individualized effect estimation
        ratio = std(dgp_dy(1:N,1:(size(train_x,2)/2))) ./ std(mu_GMM_avg);
        mu_GMM_avg = mu_GMM_avg .* ratio;
        sigma_GMM_avg = sigma_GMM_avg .* ratio; 
        D = size(mu_GMM_avg,1)*size(train_x,2)/2;
        ind_effects = dgp_dy(idx_test,1:(size(train_x,2)/2));
        results = array2table(zeros(D,3),'VariableNames',...
            {'mean','std','effect'});
        results.policy = repmat(string(policy_name),[D 1]);

        results(:,1) = num2cell(reshape(mu_GMM_avg,[D,1]));
        results(:,2) = num2cell(reshape(sigma_GMM_avg,[D,1]));
        results(:,3) = num2cell(reshape(ind_effects,[D,1]));

        disp(HYP);
        writetable(results,"./results2/ind_"+HYP+".csv");
   end
end

HYP = data_name + "_N" + int2str(N) + "_" + policy_name + "_SEED" + int2str(SEED);
csvwrite("./results2/ACC_"+HYP+".csv", ACC');

function idx_cur = softmax(IG, BATCH_SIZE)
    p = exp(IG)./sum(exp(IG));
    p_cdf = [0; cumsum(p)];
    idx_cur = zeros(BATCH_SIZE,1);
    for i=1:BATCH_SIZE
        r = rand;
        idx_cur(i) = find(r>p_cdf, 1, 'last');
    end
    idx_cur = sort(idx_cur);
end

function idx_cur = epsilon_greedy(IG, BATCH_SIZE, epsilon)
    r = rand;
    if r>=epsilon
        [~,idx_cur]=maxk(IG,BATCH_SIZE);
    else
       idx_cur = policy_uniform(1:size(IG), BATCH_SIZE); 
    end
end
