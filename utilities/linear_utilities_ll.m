function [ll,g] = linear_utilities_ll(w, train_x, train_y)
% log likelihood of linear utility model
    D = (size(train_x,2))/2;
    w = reshape(w, [],1);
    x1 = train_x(:,1:D); x2 =train_x(:,(1+D):end);
    x = [x1;x2]; % turn horizontal (x1,x2) pairs to vertical [x1;x2] 
    n = size(x1,1);
    f = x*w;
    f1 = f(1:n); f2 = f((n+1):end);
    p = normcdf((f1-f2));
    train_y = (train_y+1)/2;
    ll = mean(train_y.*log(p+1e-12) + (1-train_y).*log(1-p+1e-12)); % - w'*w/n;
    g = train_y.*normpdf(f1-f2).*(x1-x2)./(p+1e-12)...
        - (1-train_y).*normpdf(f1-f2).*(x1-x2)./(1-p+1e-12);
    g = mean(g);  % - 2*w / n
    ll = -ll;
    g = -g;
end

