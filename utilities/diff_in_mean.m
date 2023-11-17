% difference-in-mean estimator with complete independent assumption
dim_mu = [];
dim_std = [];
n_bootstrap = 1000;

for j=1:d
    if strcmp(data_name,"twoDplane")
       % is categorical
       tmp = unique(raw_x(:,j));
       vs = sort(tmp);
       for k=2:numel(vs)
           tmp1 = (1+train_y(raw_x(:,j)==vs(k-1)))./2;
           tmp2 = (1+train_y(raw_x(:,j)==vs(k)))./2;
           dim_mu = [dim_mu, (mean(tmp2)-mean(tmp1))./(vs(k)-vs(k-1))];
%            dim_std = [dim_std, sqrt(var(tmp1)/numel(tmp1)+var(tmp2)/numel(tmp2))./(vs(k)-vs(k-1))];
           dim_std = [dim_std, block_bootstrap(n_bootstrap, tmp1, tmp2)];
       end
    else
        % transform continuous to categorical
        for k=1:(BIN-1)
            lb = (k-1)/BIN; mb = (k-0)/BIN; ub = (k+1)/BIN;
            tmp1 = train_y(raw_x(:,j)>=lb & raw_x(:,j)<mb);
            tmp2 = train_y(raw_x(:,j)>=mb & raw_x(:,j)<ub);
            dim_mu = [dim_mu, (mean(tmp2)-mean(tmp1))];
%             dim_std = [dim_std, sqrt(var(tmp1)/numel(tmp1)+var(tmp2)/numel(tmp2))];
            dim_std = [dim_std, block_bootstrap(n_bootstrap, tmp1, tmp2)];
        end
    end
end

function [bootstrap_std] = block_bootstrap(n_bootstrap, data1, data2)
    if size(data1,1)<=1
        sample1 = data1.*ones(n_bootstrap,1);
    else
        sample1 = bootstrp(n_bootstrap,@mean,data1);
    end
    if size(data2,1)<=1
        sample2 = data2.*ones(n_bootstrap,1);
    else
        sample2 = bootstrp(n_bootstrap,@mean,data2);
    end
    bootstrap_std = std(sample1 - sample2);
end