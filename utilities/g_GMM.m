function [mu_GMM_avg,sigma_GMM_avg, mu_GMM,sigma_GMM, dy_mu, dy_std, df_mu, df_K, ks, ws] = g_GMM(n_gauss_hermite,hyp,...
    inffunc,meanfunc,covfunc, likfunc, train_x, train_y, test_x) 
% Approximate dp(y=1)/dx with GMM
% 
% Copyright (c) by Yehu Chen, 2023-06-19.
%
    D = size(train_x,2)/2;
   [~,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, ...
        covfunc, likfunc, train_x, train_y, test_x);
    
    % clip to range of -3 to 3
%     fmu(fmu<=-3) = -3;
%     fmu(fmu>=3) = 3;

    % compute marginal effects of infinitesimal 
    % change at leftside comparison                                              
    dK = dcovPref(covfunc{2},hyp,test_x,train_x);
    df_mu = zeros(size(test_x,1),D);
    for j=1:D
        df_mu(:,j) = dK(:,:,j)*post.alpha;
%         if strcmp(functions(meanfunc{1}).function,'meanLinear')
%             df_mu(:,j) = df_mu(:,j) + hyp.mean(j);
%         end
    end

    d2K = d2covPref(covfunc{2},hyp,test_x,test_x);
    df_K = zeros(size(d2K));
    for i=1:size(test_x,1)
        tmp = repmat(post.sW,1,D).*squeeze(dK(i,:,:));
        tmp = post.L'\tmp;
        df_K(i,:,:) = squeeze(d2K(i,:,:)) - diag(sum(tmp.*tmp,1));
    end

    dy_mu = zeros(size(test_x,1),D);
    for j=1:D
        dy_mu(:,j) = df_mu(:,j).*normpdf(fmu);
    end

    dy_std = zeros(size(test_x,1),D);
    for j=1:D
        dy_std(:,j) = sqrt(df_K(:,j,j)).*normpdf(fmu);
    end
    
    % clip to at least 0.01 for numerical stability
    dy_std(dy_std<=0.01) = 0.01;

    % GMM for dy
    [ks,ws] = root_GH(n_gauss_hermite);

    mu_GMM = zeros(size(test_x,1), D, n_gauss_hermite);
    sigma_GMM = zeros(size(test_x,1), D, n_gauss_hermite);
    
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
    mu_GMM_avg = zeros(size(test_x,1), D);
    sigma_GMM_avg = zeros(size(test_x,1), D);

    for i=1:size(test_x,1)
        mu_GMM_avg(i,:) = squeeze(mu_GMM(i,:,:)) * ws;
        sigma_GMM_avg(i,:) = sqrt(squeeze(sigma_GMM(i,:,:).^2) * ws + ...
            squeeze(mu_GMM(i,:,:).^2) * ws - ...
            mu_GMM_avg(i,:)'.^2);
    end
end