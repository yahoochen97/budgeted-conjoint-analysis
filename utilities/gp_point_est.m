% gp preference learning point estimation effect
function [gp_point_mu,gp_point_std]=gp_point_est(BIN,raw_x,dy_mu,dy_std)
    gp_point_mu = [];
    gp_point_std = [];
    d = size(raw_x,2) / 2;
    for j=1:d
        tmp = unique(raw_x(:,j));
        if floor(tmp(1))==tmp(1)
           % is categorical
           vs = sort(tmp);
           for k=2:numel(vs)
               tmp1 = dy_mu(raw_x(:,j)==vs(k),1+size(gp_point_mu,2));
               gp_point_mu = [gp_point_mu, mean(tmp1)];
               gp_point_std = [gp_point_std, std(tmp1)/sqrt(numel(tmp1))];
           end
        else
            % transform continuous to categorical
            for k=1:(BIN-1)
                lb = (k-1)/BIN; mb = k/BIN; ub = (k+1)/BIN;
                tmp1 = dy_mu(raw_x(:,j)>=lb & raw_x(:,j)<mb, 1+size(gp_point_mu,2));
                tmp2 = dy_mu(raw_x(:,j)>=mb & raw_x(:,j)<ub, 1+size(gp_point_mu,2));
                gp_point_mu = [gp_point_mu, (mean(tmp2)+mean(tmp1))/2];
                tmp1 = dy_std(raw_x(:,j)>=lb & raw_x(:,j)<mb, 1+size(gp_point_mu,2));
                tmp2 = dy_std(raw_x(:,j)>=mb & raw_x(:,j)<ub, 1+size(gp_point_mu,2));
                gp_point_std = [gp_point_std, sqrt(tmp1'*tmp1/numel(tmp1)+tmp2'*tmp2/numel(tmp2))/2];
            end
        end
    end
end