% difference-in-mean estimator with complete independent assumption
dim_mu = [];
dim_std = [];
for j=1:d
    tmp = unique(raw_x(:,j));
    if floor(tmp(1))==tmp(1)
       % is categorical
       vs = sort(tmp);
       for k=2:numel(vs)
           tmp1 = (1+train_y(raw_x(:,j)==vs(k-1)))./2;
           tmp2 = (1+train_y(raw_x(:,j)==vs(k)))./2;
           dim_mu = [dim_mu, (mean(tmp2)-mean(tmp1))./(vs(k)-vs(k-1))];
           dim_std = [dim_std, sqrt(var(tmp1)/numel(tmp1)+var(tmp2)/numel(tmp2))./(vs(k)-vs(k-1))];
       end
    else
        % transform continuous to categorical
        for k=1:(BIN-1)
            lb = (k-1)/BIN; mb = (k+0)/BIN; ub = (k+1)/BIN;
            tmp1 = train_y(raw_x(:,j)>=lb & raw_x(:,j)<mb);
            tmp2 = train_y(raw_x(:,j)>=mb & raw_x(:,j)<ub);
            dim_mu = [dim_mu, (mean(tmp2)-mean(tmp1))];
            dim_std = [dim_std, sqrt(var(tmp1)/numel(tmp1)+var(tmp2)/numel(tmp2))];
        end
    end
end