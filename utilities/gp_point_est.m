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
               tmp2 = dy_mu(raw_x(:,j)==vs(k-1),1+size(gp_point_mu,2));
               tmp3 = dy_std(raw_x(:,j)==vs(k),1+size(gp_point_mu,2));
               tmp4 = dy_std(raw_x(:,j)==vs(k-1),1+size(gp_point_mu,2));
%                if numel(tmp1)==0, tmp1 = tmp2; end
%                if numel(tmp2)==0, tmp2 = tmp1; end
%                if numel(tmp3)==0, tmp3 = tmp4; end
%                if numel(tmp4)==0, tmp4 = tmp3; end
               gp_point_mu = [gp_point_mu, (mean(tmp1)+mean(tmp2))/2];
               v1 = (mean(tmp3.^2) + mean(tmp1.^2) - mean(tmp1)^2)/numel(tmp1);% + 
               v2 = (mean(tmp4.^2) + mean(tmp2.^2) - mean(tmp2)^2)/numel(tmp2);% + 
               gp_point_std = [gp_point_std, sqrt(v1+v2)/2];
           end
        else
            % transform continuous to categorical
            for k=1:(BIN-1)
                lb = (k-1)/BIN; mb = k/BIN; ub = (k+1)/BIN;
                tmp1 = dy_mu(raw_x(:,j)>=lb & raw_x(:,j)<mb, j);
                tmp2 = dy_mu(raw_x(:,j)>=mb & raw_x(:,j)<ub, j);
%                 if numel(tmp1)==0, tmp1 = tmp2; end
%                 if numel(tmp2)==0, tmp2 = tmp1; end
                gp_point_mu = [gp_point_mu, (mean(tmp1)+mean(tmp2))/2];
                tmp3 = dy_std(raw_x(:,j)>=lb & raw_x(:,j)<mb, j);
                tmp4 = dy_std(raw_x(:,j)>=mb & raw_x(:,j)<ub, j);
%                 if numel(tmp3)==0, tmp3 = tmp4; end
%                 if numel(tmp4)==0, tmp4 = tmp3; end
                v1 = (mean(tmp3.^2) + mean(tmp1.^2) - mean(tmp1)^2)/numel(tmp1);% + 
                v2 = (mean(tmp4.^2) + mean(tmp2.^2) - mean(tmp2)^2)/numel(tmp2);
                gp_point_std = [gp_point_std, sqrt(v1+v2)/2];
            end
        end
    end
end