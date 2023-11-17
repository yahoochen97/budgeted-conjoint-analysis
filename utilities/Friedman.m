function [y, p, f, df, dy] = Friedman(pair_x)
    % simulate data from Friedman described in
    % https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html
    %
    % p(y=1) = Phi(f1-f2) indicating x1 is preferred to x2
    % dy = [dp(y=1)/dx1, dp(y=1)/dx2]
    %
    % inputs:
    %       -x: n x 2d matrix of n pairs of (x1,x2)
    % outputs:
    %       -y: n x 1 vector of preferences
    %       -p: n x 1 vector of prob of preferences
    %       -f: n x 2 matrix of n pairs of latent utilities
    %       -df: n x 2d matrix of n pairs of gradients in utilities
    %       -dy: n x 2d vector of n pairs of gradients in prob of
    %       preferences
    
    x1 = pair_x(:,1:end/2); x2 = pair_x(:,1+end/2:end);
    x = [x1;x2]; % turn horizontal (x1,x2) pairs to vertical [x1;x2] 
    n = size(x1,1);
    f = sin(pi*x(:,1).*x(:,2)) + 2*(x(:,3)-0.5).^2;
    f1 = f(1:n); f2 = f((n+1):end);
    f = [f1,f2];
    p = normcdf((f1-f2));
    y = 2*arrayfun(@(x) binornd(1,x),p)-1; % y in {-1,1}
    
    df = zeros(2*n,size(x1,2)); % df/dx evaluated at x
    df(:,1) = cos(pi*x(:,1).*x(:,2)).*x(:,2)*pi;
    df(:,2) = cos(pi*x(:,1).*x(:,2)).*x(:,1)*pi;
    df(:,3) = 4*(x(:,3)-0.5);
    df1 = df(1:n,:); df2 = df((n+1):end,:);
    df = [df1,df2];
    
    % gradient of prob
    dy = [df1.*normpdf((f1-f2)/1), -df2.*normpdf((f1-f2)/1)];
    dy = round(dy,4);
end