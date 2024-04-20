% gp preference learning averaged marginal component effects
function [gp_mu,gp_std]=gp_AMCE(dy_mu,dy_std, data_name, train_x)
% compute averaged marginal component effect for each compoment
% given the estimated gradients of transformed attributes for population
    d = size(train_x,2)/2;
    N = size(dy_mu,1);
    gp_mu = [];
    gp_std = [];
    for j=1:d
        if strcmp(data_name,"Friedman")
            BIN = 1;
            for k=1:BIN
                lb = (k-1)/BIN; mb = (k-0)/BIN; % ub = (k+1)/BIN;
                tmp1 = train_x(:,j)>=lb & train_x(:,j)<mb;
%                 tmp2 = train_x(:,j)>=mb & train_x(:,j)<ub;
%                 gp_mu = [gp_mu, (mean(dy_mu(tmp1,j))+mean(dy_mu(tmp2,j)))/2];
                gp_mu = [gp_mu, mean(dy_mu(tmp1,j))];
                var1 = mean(dy_std(tmp1,j).^2)/sum(tmp1) + var(dy_mu(tmp1,j))/sum(tmp1);
%                 var2 = mean(dy_std(tmp2,j).^2)/sum(tmp2);
                gp_std = [gp_std, sqrt(var1)];
            end
        else
            gp_mu = [gp_mu, mean(dy_mu(:,j))];
            gp_std = [gp_std, sqrt(mean(dy_std(:,j).^2)/N + var(dy_mu(:,j))/N)];
        end
    end
end