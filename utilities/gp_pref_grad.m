% build a gp preference learning model for grad
% initialize gp model
D = size(train_x,2)/2;
meanfunc = {@meanZero};  
covfunc = {@covPref, {@covSum,{@covSEard, @covLINiso}}};   % SEard + LINiso           
likfunc = {@likErf};
hyp.mean = [];
hyp.cov = [zeros(D,1);log(4); log(1)];

% assign prior for length scales
for i=1:D
    prior.cov{i} = {@priorTransform,@exp,@exp,@log,{@priorInvGauss,1,2}};
end

% slope variance: fix to 1
prior.cov{D+2} = {@priorDelta};

% output scale
if strcmp(data_name,"Friedman")
%     prior.cov{D+1} = {@priorTransform,@exp,@exp,@log,{@priorInvGauss,3,1}};
    prior.cov{D+1} = {@priorDelta};
    hyp.cov(D+1) = log(1);
    inffunc = {@infPrior, @infEP, prior};
end

if strcmp(data_name,"twoDplane")
    prior.cov{D+1} = {@priorDelta};
    hyp.cov(D+1) = log(4);
    inffunc = {@infPrior, @infLaplace, prior};
end

if strcmp(data_name,"hainmueller_candidate") || strcmp(data_name,"hainmueller_immigrant")
    prior.cov{D+1} = {@priorDelta};
    hyp.cov(D+1) = log(4);
    inffunc = {@infPrior, @infLaplace, prior};
end

% elseif strcmp(data_name,"Friedman")
%     prior.cov{D+1} = {@priorTransform,@exp,@exp,@log,{@priorInvGauss,2,4}};
%     inffunc = {@infPrior, @infEP, prior};
% else
%     % application
%     prior.cov{D+1} = {@priorTransform,@exp,@exp,@log,{@priorInvGauss,2,4}};
%     inffunc = {@infPrior, @infEP, prior};
% end

p.method = 'LBFGS';
p.length = 100;

% optimize hyperparameters and inference test locations
if learn_HYP
    hyp = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, train_x, train_y);
end
[ymu,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, ...
    covfunc, likfunc, train_x, train_y, test_x);
% 
% % visualize latent utilities for test locations
% % fig = figure(1);
% % scatter(dgp_f(:,1)-dgp_f(:,2),fmu);
% % xlabel("dgp f(x1)-f(x2)", 'FontSize',FONTSIZE);
% % ylabel("estimated f(x1)-f(x2)",'FontSize',FONTSIZE);
% 
% % compute marginal effects of infinitesimal 
% % change at leftside comparison                                              
% dK = dcovPref(covfunc{2},hyp,test_x,train_x);
% df_mu = zeros(size(test_x,1),D);
% for j=1:D
%     df_mu(:,j) =  dK(:,:,j)*post.alpha;
% end
% 
% d2K = d2covPref(covfunc{2},hyp,test_x,test_x);
% df_K = zeros(size(d2K));
% for i=1:size(test_x,1)
%     tmp = repmat(post.sW,1,D).*squeeze(dK(i,:,:));
%     tmp = post.L'\tmp;
%     df_K(i,:,:) = squeeze(d2K(i,:,:)) - diag(sum(tmp.*tmp,1));
% end
% 
% % true df(x1)/dx1
% % fig = figure(2);
% % scatter(dgp_df(:,2),df_mu(:,2));
% % errorbar(dgp_df(:,2),df_mu(:,2),sqrt(df_K(:,2,2)),'o');
% % 
% % xlabel("dgp df(x2)", 'FontSize',FONTSIZE);
% % ylabel("estimated df(x2)",'FontSize',FONTSIZE);
% 
% % true p(y=1)
% % fig = figure(3);
% % scatter(dgp_p,(ymu+1)/2);
% % 
% % xlabel("true p(y=1)", 'FontSize',FONTSIZE);
% % ylabel("estimated p(y=1)",'FontSize',FONTSIZE);
% 
% % true dy(x1)/dx1
% % fig = figure(4);
% dy_mu = zeros(size(test_x,1),D);
% for j=1:D
%     dy_mu(:,j) = df_mu(:,j).*normpdf(fmu);
% end
% 
% dy_std = zeros(size(test_x,1),D);
% for j=1:D
%     dy_std(:,j) = sqrt(df_K(:,j,j)).*normpdf(fmu);
% end
% 
% % scatter(dgp_dy(:,2),dy_mu(:,2));
% % errorbar(dgp_dy(:,2),dy_mu(:,2),sqrt(df_K(:,2,2)),'o');
% % 
% % xlabel("dgp dy(x2)", 'FontSize',FONTSIZE);
% % ylabel("estimated dy(x2)",'FontSize',FONTSIZE);
% 
% 
% % GMM for dy
% n_gauss_hermite = 10;
% [ks,ws] = root_GH(n_gauss_hermite);
% 
% mu_GMM = zeros(size(test_x,1), D, n_gauss_hermite);
% sigma_GMM = zeros(size(test_x,1), D, n_gauss_hermite);
% 
% for i=1:size(test_x,1)
%     dmus = df_mu(i,:);
%     dsigmas = squeeze(df_K(i,:,:));
%     
%     for k=1:n_gauss_hermite
%         f_bar = sqrt(2)*ks(k)*sqrt(fs2(i)) + fmu(i);
%         mu_GMM(i,:,k) = normpdf(f_bar).*dmus;
%         K_bar = (normpdf(f_bar)*normpdf(f_bar)').*dsigmas;
%         sigma_GMM(i,:,k) = sqrt(diag(K_bar));
%     end
% end

[mu_GMM_avg,sigma_GMM_avg, mu_GMM,sigma_GMM, dy_mu, dy_std, df_mu, df_K, ks, ws] = g_GMM(n_gauss_hermite,hyp,inffunc,meanfunc,covfunc, likfunc, train_x, train_y, test_x);


