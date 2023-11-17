function [y, p, f, df, dy] = twoDplane(pair_x)
    % simulate data from 2Dplane described in
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
    f = zeros(2*n,1);
    % f(x(:,1)==1) =  3*x(x(:,1)==1,2) - 2*x(x(:,1)==1,3) + x(x(:,1)==1,4);
    % f(x(:,1)==-1) =  - 3*x(x(:,1)==-1,5) + 2*x(x(:,1)==-1,6) - x(x(:,1)==-1,7);
    f(x(:,1)==1) =  x(x(:,1)==1,2:3)*[2;2] + x(x(:,1)==1,4:5)*[-1;-1];
    f(x(:,1)==0) =  x(x(:,1)==0,6:7)*[1;1] + x(x(:,1)==0,8:9)*[-2;-2];
    f = f + 2*x(:,1);
    f1 = f(1:n); f2 = f((n+1):end);
    f = [f1,f2];
    p = normcdf((f1-f2));
    y = 2*arrayfun(@(x) binornd(1,x),p)-1; % y in {-1,1}
    
    df = zeros(2*n,size(x1,2)); % df/dx evaluated at x
    df(:,1) = 2;
    df(x(:,1)==1,2:3) = 2;
    df(x(:,1)==1,4:5) = -1;
    df(x(:,1)==0,6:7) = 1;
    df(x(:,1)==0,8:9) = -2;
    df1 = df(1:n,:); df2 = df((n+1):end,:);
    df = [df1,df2];
    
    % gradient of prob
    dy = [df1.*normpdf((f1-f2)/1), -df2.*normpdf((f1-f2)/1)];
    dy = round(dy,4);
end