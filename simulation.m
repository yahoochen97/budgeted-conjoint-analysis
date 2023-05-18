% add gpml tool box; clear env;
clear;
close all;
addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
addpath("utilities");
FONTSIZE=16;

% simulate data from 2D plane
% https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html
n = 100;
d = 4;
x = zeros(n,d);
x(:,1) = 2*randi([0 1],n,1)-1; % p(x=-1)=p(x=1)=1/2

for j=2:d
    x(:,j) = randi([-1 1],n,1); % p(x=-1)=p(x=0)=p(x=1)=1/3
end

N = 1000;
pair_idx = nchoosek(1:n, 2);
pair_idx = pair_idx(randperm(size(pair_idx,1),N)',:);
pair_x = [x(pair_idx(:,1),:), x(pair_idx(:,2),:)];
[pair_y, p_pref, f, df, dy] = twoDplanes(pair_x);

% split train/test data
train_x = pair_x(1:(N*4/5),:);
train_y = pair_y(1:(N*4/5),:);
test_x = pair_x((N*4/5+1):N,:);

% build a gp model
meanfunc = {@meanZero};   
covfunc = {@covPref, {@covSEard}};             
likfunc = {@likErf};
inffunc = {@infEP};
hyp.mean = [];
hyp.cov = [zeros(d,1);log(1)];

for i=1:d
   prior.cov{i} = {@priorTransform,@exp,@exp,@log,{@priorGamma,1,1}};
end

prior.cov{d+1} = {@priorDelta};

inffunc = {@infPrior, @infEP, prior};
p.method = 'LBFGS';
p.length = 100;

% hyp = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, train_x, train_y);
[ymu,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, ...
    covfunc, likfunc, train_x, train_y, test_x);

% visualize latent utilities for test locations
fig = figure(1);
scatter(f((N*4/5+1):N,1)-f((N*4/5+1):N,2),fmu);
xlabel("true f(x1)-f(x2)", 'FontSize',FONTSIZE);
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
df_dgp = df((N*4/5+1):end,1:d);

fig = figure(2);
scatter(df_dgp(:,2),df_mu(:,2));
% errorbar(df_dgp(:,2),df_mu(:,2),df_K(:,2,2),'o');

xlabel("true df(x1)", 'FontSize',FONTSIZE);
ylabel("estimated df(x1)",'FontSize',FONTSIZE);

% true df(x1)/dx1
fig = figure(3);
dy_mu = zeros(size(test_x,1),d);
for j=1:d
    dy_mu(:,j) = df_mu(:,j).*((ymu+1)/2);
end

dy_dgp = dy((N*4/5+1):end,1:d);
scatter(dy_dgp(:,3),dy_mu(:,3));

xlabel("true dy(x1)", 'FontSize',FONTSIZE);
ylabel("estimated dy(x1)",'FontSize',FONTSIZE);

% true p(y=1)
fig = figure(4);
scatter(p_pref((N*4/5+1):end),(ymu+1)/2);

xlabel("true p(y=1)", 'FontSize',FONTSIZE);
ylabel("estimated p(y=1)",'FontSize',FONTSIZE);

% GMM for gradient of prob
n_gauss_hermite = 5;
[ks,ws] = root_GH(n_gauss_hermite);

fig = figure(5);
scatter(dy_dgp(:,3),dy_mu(:,3));
sigma_GMM = zeros(size(test_x,1),d);

for i=1:size(test_x,1)
    dmus = df_mu(i,:);
    dsigmas = squeeze(df_K(i,:,:));
         
%     mu_GMM = {};
%     sigma_GMM = {};
    
    for k=1 % 1:n_gauss_hermite
        f_bar = sqrt(2)*ks(k)*sqrt(fs2(i)) + fmu(i);
        mu_bar = normpdf(f_bar).*dmus;
        K_bar = (normpdf(f_bar)*normpdf(f_bar)').*dsigmas;
        sigma_GMM(i,:) = sqrt(diag(K_bar));
%         mu_GMM{i} = mu_bar;
%         sigma_GMM{k} = K_bar;
    end
end

errorbar(dy_dgp(:,3),dy_mu(:,3),sigma_GMM(:,3),'o');

