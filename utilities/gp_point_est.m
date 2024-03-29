% gp preference learning point estimation effect
function [gp_point_mu,gp_point_std]=gp_point_est(BIN,raw_x,dy_mu,dy_std)
    gp_point_mu = [];
    gp_point_std = [];
    d = size(raw_x,2) / 2;
    for j=1:d
        tmp = unique(raw_x(:,j));
        if floor(tmp(1))==tmp(1)
           % is categorical
           if j==1
               vs = [-1,1];
           else
               vs = [-1,0,1];
           end
           for k=2:numel(vs)
               tmp1 = dy_mu(raw_x(:,j)==vs(k),1+size(gp_point_mu,2));
               tmp2 = dy_mu(raw_x(:,j)==vs(k-1),1+size(gp_point_mu,2));
               tmp3 = dy_std(raw_x(:,j)==vs(k),1+size(gp_point_mu,2));
               tmp4 = dy_std(raw_x(:,j)==vs(k-1),1+size(gp_point_mu,2));
               tmp1 = [tmp1;tmp2];
               tmp3 = [tmp3;tmp4];
               gp_point_mu = [gp_point_mu, (mean(tmp1))/1];
               v1 = var(tmp1)/numel(tmp1) + mean(tmp3.^2)/numel(tmp3);
               v2 = var(tmp2)/numel(tmp2) + mean(tmp4.^2)/numel(tmp4);
               gp_point_std = [gp_point_std, sqrt(v1)/1];
           end
        else
            % transform continuous to categorical
            for k=1:(BIN)
                lb = (k-1)/BIN; mb = (k-0)/BIN; ub = (k+1)/BIN;
                tmp1 = dy_mu(raw_x(:,j)>=lb & raw_x(:,j)<mb, j);
                tmp2 = dy_mu(raw_x(:,j)>=mb & raw_x(:,j)<ub, j);
                gp_point_mu = [gp_point_mu, (mean(tmp1))/1];
                tmp3 = dy_std(raw_x(:,j)>=lb & raw_x(:,j)<mb, j);
                tmp4 = dy_std(raw_x(:,j)>=mb & raw_x(:,j)<ub, j);
                v1 = var(tmp1)/numel(tmp1) + mean(tmp3.^2)/numel(tmp3); 
                v2 = var(tmp2)/numel(tmp2) + mean(tmp4.^2)/numel(tmp4);
                gp_point_std = [gp_point_std, sqrt(v1)];
            end
        end
    end
end