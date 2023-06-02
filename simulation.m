if ~exist('SEED','var')
    % simulation settings
    SEED = 1;
    d = 7;
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

% simulate profile data from 2D plane (categorical attributes)
% https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html
% x = zeros(n,d);
% x(:,1) = 2*randi([0 1],n,1)-1; % p(x=-1)=p(x=1)=1/2
% 
% for j=2:d
%     x(:,j) = randi([-1 1],n,1); % p(x=-1)=p(x=0)=p(x=1)=1/3
% end

% randomly select pairs from profiles
% pair_idx = nchoosek(1:n, 2);
% pair_idx = pair_idx(randperm(size(pair_idx,1),N)',:);
% pair_idx = zeros(N,2);
% for i = 1:N
%     pair_idx(i,:) = randperm(n,2) ; 
% end
% pair_x = [x(pair_idx(:,1),:), x(pair_idx(:,2),:)];
pair_x = [2*randi([0 1],N,1)-1,randi([-1 1],N,d-1),2*randi([0 1],N,1)-1,randi([-1 1],N,d-1)];
% pair_x = 2*randi([0 1],N,2*d)-1;
[pair_y, dgp_p, dgp_f, dgp_df, dgp_dy] = twoDplanes(pair_x);

% split train/test data
% train_ratio = 4/5;
% train_x = pair_x(1:(N*train_ratio),:);
% train_y = pair_y(1:(N*train_ratio),:);
% test_x = pair_x((N*train_ratio+1):N,:);
% test_y = pair_y((N*train_ratio+1):N,:);
train_ratio = 1;
train_x = pair_x;
train_y = pair_y;
test_x = train_x;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% true marginal effect
dgp_effects = zeros(3*d-2,1); % 1 for dim 1, 2 for dim 2...d

dgp_effects(1) = mean(dgp_dy(train_x(:,1)==1,1));
dgp_effects(2) = mean(dgp_dy(train_x(:,1)==-1,1));
for j=2:d
    dgp_effects(3*j-3) = mean(dgp_dy(train_x(:,j)==1,j));
    dgp_effects(3*j-2) = mean(dgp_dy(train_x(:,j)==0,j));
    dgp_effects(3*j-1) = mean(dgp_dy(train_x(:,j)==-1,j));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% difference-in-mean estimator with complete independent assumption
dim_mu = zeros(2*d-1,1); % 1 for dim 1, 2 for dim 2...d
dim_std = zeros(2*d-1,1);

dim_mu(1) = (mean(train_y(train_x(:,1)==1)) - mean(train_y(train_x(:,1)==-1)))/2;
tmp1 = train_y(train_x(:,1)==1);
tmp2 = train_y(train_x(:,1)==-1);
dim_std(1) = sqrt(var(tmp1)/size(tmp1,1) + var(tmp2)/size(tmp2,1))/2;
for j=2:d
    dim_mu(2*j-2) = mean(train_y(train_x(:,j)==1)) - mean(train_y(train_x(:,j)==-1));
    tmp1 = train_y(train_x(:,j)==1);
    tmp2 = train_y(train_x(:,j)==0);
    dim_std(2*j-2) = sqrt(var(tmp1)/size(tmp1,1) + var(tmp2)/size(tmp2,1));
    
    dim_mu(2*j-1) = mean(train_y(train_x(:,j)==0)) - mean(train_y(train_x(:,j)==-1));
    tmp1 = train_y(train_x(:,j)==0);
    tmp2 = train_y(train_x(:,j)==-1);
    dim_std(2*j-1) = sqrt(var(tmp1)/size(tmp1,1) + var(tmp2)/size(tmp2,1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build a gp model
meanfunc = {@meanZero};   
covfunc = {@covPref, {@covSEard}};             
likfunc = {@likErf};
% inffunc = {@infEP};
hyp.mean = [];
hyp.cov = [zeros(d,1);log(1)];

% assign prior for length scales
for i=1:d
   prior.cov{i} = {@priorTransform,@exp,@exp,@log,{@priorInvGauss,1,10}};
end

% output scale
prior.cov{d+1} = {@priorTransform,@exp,@exp,@log,{@priorInvGauss,2,4}};

inffunc = {@infPrior, @infEP, prior};
p.method = 'LBFGS';
p.length = 100;

% optimize hyperparameters and inference test locations
hyp = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, train_x, train_y);
[ymu,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, ...
    covfunc, likfunc, train_x, train_y, test_x);

% visualize latent utilities for test locations
fig = figure(1);
scatter(dgp_f(:,1)-dgp_f(:,2),fmu);
xlabel("dgp f(x1)-f(x2)", 'FontSize',FONTSIZE);
ylabel("estimated f(x1)-f(x2)",'FontSize',FONTSIZE);

% compute marginal effects of infinitesimal 
% change at leftside comparison                                              
dK = dcovPref(covfunc{2},hyp,test_x,train_x);
df_mu = zeros(size(test_x,1),d);
for j=1:d
    df_mu(:,j) =  dK(:,:,j)*post.alpha;
end

d2K = d2covPref(covfunc{2},hyp,test_x,test_x);
df_K = zeros(size(d2K));
for i=1:size(test_x,1)
    tmp = repmat(post.sW,[1 d]).*squeeze(dK(i,:,:));
    tmp = post.L'\tmp;
    df_K(i,:,:) = squeeze(d2K(i,:,:)) - tmp'*tmp;
end

% true df(x1)/dx1
fig = figure(2);
scatter(dgp_df(:,2),df_mu(:,2));
% errorbar(dgp_df(:,2),df_mu(:,2),sqrt(df_K(:,2,2)),'o');

xlabel("dgp df(x2)", 'FontSize',FONTSIZE);
ylabel("estimated df(x2)",'FontSize',FONTSIZE);

% true p(y=1)
fig = figure(3);
scatter(dgp_p,(ymu+1)/2);

xlabel("true p(y=1)", 'FontSize',FONTSIZE);
ylabel("estimated p(y=1)",'FontSize',FONTSIZE);

% true dy(x1)/dx1
fig = figure(4);
dy_mu = zeros(size(test_x,1),d);
for j=1:d
    dy_mu(:,j) = df_mu(:,j).*normpdf(fmu);
end

scatter(dgp_dy(:,2),dy_mu(:,2));
errorbar(dgp_dy(:,2),dy_mu(:,2),sqrt(df_K(:,2,2)),'o');

xlabel("dgp dy(x2)", 'FontSize',FONTSIZE);
ylabel("estimated dy(x2)",'FontSize',FONTSIZE);

% point estimation effect
est_effects_point = zeros(3*d-2,2); % 1 for dim 1, 2 for dim 2...d

est_effects_point(1,1) = mean(dy_mu(train_x(:,1)==1,1));
est_effects_point(2,1) = mean(dy_mu(train_x(:,1)==-1,1));
est_effects_point(1,2) = std(dy_mu(train_x(:,1)==1,1));
est_effects_point(2,2) = std(dy_mu(train_x(:,1)==-1,1));
for j=2:d
    est_effects_point(3*j-3,1) = mean(dy_mu(train_x(:,j)==1,j));
    est_effects_point(3*j-2,1) = mean(dy_mu(train_x(:,j)==0,j));
    est_effects_point(3*j-1,1) = mean(dy_mu(train_x(:,j)==-1,j));
    est_effects_point(3*j-3,2) = std(dy_mu(train_x(:,j)==1,j));
    est_effects_point(3*j-2,2) = std(dy_mu(train_x(:,j)==0,j));
    est_effects_point(3*j-1,2) = std(dy_mu(train_x(:,j)==-1,j));
end

fig = figure(5);
scatter(dgp_effects,est_effects_point(:,1))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM for dy
n_gauss_hermite = 5;
[ks,ws] = root_GH(n_gauss_hermite);

mu_GMM = zeros(size(test_x,1), d, n_gauss_hermite);
sigma_GMM = zeros(size(test_x,1), d, n_gauss_hermite);

for i=1:size(test_x,1)
    dmus = df_mu(i,:);
    dsigmas = squeeze(df_K(i,:,:));
    
    for k=1:n_gauss_hermite
        f_bar = sqrt(2)*ks(k)*sqrt(fs2(i)) + fmu(i);
        mu_GMM(i,:,k) = normpdf(f_bar).*dmus;
        K_bar = (normpdf(f_bar)*normpdf(f_bar)').*dsigmas;
        sigma_GMM(i,:,k) = sqrt(diag(K_bar));
    end
end

% average over GMM
mu_GMM_avg = zeros(size(test_x,1), d);
sigma_GMM_avg = zeros(size(test_x,1), d);

for i=1:size(test_x,1)
    mu_GMM_avg(i,:) = squeeze(mu_GMM(i,:,:)) * ws;
    sigma_GMM_avg(i,:) = sqrt(squeeze(sigma_GMM(i,:,:).^2) * ws + ...
        squeeze(mu_GMM(i,:,:).^2) * ws - ...
        mu_GMM_avg(i,:)'.^2);
end

% fig = figure(6);
% 
% results_GMM_avg = array2table([mu_GMM_avg,sigma_GMM_avg,dgp_dy(:,1:d),test_x(:,1:d)],...
% 'VariableNames',[sprintfc('mean%d', 1:d),sprintfc('std%d', 1:d), sprintfc('dy%d', 1:d),sprintfc('x%d', 1:d)]);
% 
% scatter(results_GMM_avg.dy2,...
%     results_GMM_avg.mean2);

% GMM estimation effect
est_effects_GMM = zeros(3*d-2,2); % 1 for dim 1, 2 for dim 2...d

est_effects_GMM(1,1) = mean(mu_GMM_avg(train_x(:,1)==1,1));
est_effects_GMM(2,1) = mean(mu_GMM_avg(train_x(:,1)==-1,1));
est_effects_GMM(1,2) = std(mu_GMM_avg(train_x(:,1)==1,1));
est_effects_GMM(2,2) = std(mu_GMM_avg(train_x(:,1)==-1,1));
for j=2:d
    est_effects_GMM(3*j-3,1) = mean(mu_GMM_avg(train_x(:,j)==1,j));
    est_effects_GMM(3*j-2,1) = mean(mu_GMM_avg(train_x(:,j)==0,j));
    est_effects_GMM(3*j-1,1) = mean(mu_GMM_avg(train_x(:,j)==-1,j));
    est_effects_GMM(3*j-3,2) = std(mu_GMM_avg(train_x(:,j)==1,j));
    est_effects_GMM(3*j-2,2) = std(mu_GMM_avg(train_x(:,j)==0,j));
    est_effects_GMM(3*j-1,2) = std(mu_GMM_avg(train_x(:,j)==-1,j));
end

fig = figure(6);
scatter(dgp_effects,est_effects_GMM(:,1));

results = array2table(zeros(3*(2*d-1),3),'VariableNames',...
    {'mean','std','effect'});

results.model = cell(6*d-3,1);

true_effects = grad2marg(dgp_effects);
results(1:(2*d-1),1) = num2cell(dim_mu);
results(1:(2*d-1),2) = num2cell(dim_std);
results(1:(2*d-1),4) = {'diffinmean'};
results(1:(2*d-1),3) = num2cell(true_effects);

results((2*d):(4*d-2),1) = num2cell(grad2marg(est_effects_point(:,1)));
results((2*d):(4*d-2),2) = num2cell(grad2margstd(est_effects_point(:,2)));
results((2*d):(4*d-2),4) = {'gppoint'};
results((2*d):(4*d-2),3) = num2cell(true_effects);

results((4*d-1):(6*d-3),1) = num2cell(grad2marg(est_effects_GMM(:,1)));
results((4*d-1):(6*d-3),2) = num2cell(grad2margstd(est_effects_GMM(:,2)));
results((4*d-1):(6*d-3),4) = {'gpGMM'};
results((4*d-1):(6*d-3),3) = num2cell(true_effects);

HYP = "2Dplane_N" + int2str(N) + "_d" + int2str(d) + "_SEED" + int2str(SEED);

writetable(results,"./results/"+HYP+".csv");
