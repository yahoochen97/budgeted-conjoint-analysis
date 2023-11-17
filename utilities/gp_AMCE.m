% gp preference learning averaged marginal component effects
function [gp_mu,gp_std]=gp_AMCE(dy_mu,dy_std, data_name, train_x)
% compute averaged marginal component effect for each compoment
% given the estimated gradients of transformed attributes for population
    d = size(train_x,2)/2;
    N = size(dy_mu,1);
    gp_mu = [];
    gp_std = [];
    BIN = 10;
    for j=1:d
        if strcmp(data_name,"twoDplane")
            gp_mu = [gp_mu, mean(dy_mu(:,j))];
            gp_std = [gp_std, sqrt(sum(dy_std(:,j).^2))./N];
        else
            for k=1:(BIN-1)
                lb = (k-1)/BIN; mb = (k-0)/BIN; ub = (k+1)/BIN;
                tmp1 = train_x(:,j)>=lb & train_x(:,j)<mb;
                tmp2 = train_x(:,j)>=mb & train_x(:,j)<ub;
                gp_mu = [gp_mu, (mean(dy_mu(tmp1,j))+mean(dy_mu(tmp2,j)))/2];
                gp_std = [gp_std, sqrt(sum([dy_std(tmp1,j);dy_std(tmp2,j)].^2))./(sum(tmp1)+sum(tmp2))];
            end
        end
    end
end