D = (size(train_x,2))/2;
w = zeros(D,1);
% check grad 
% for i=1:D
%     jitter = zeros(D,1);
%     jitter(i) = 1e-6;
%     [ll1,g] = linear_utilities_ll(w, train_x, train_y);
%     [ll2,~] = linear_utilities_ll(w+jitter, train_x, train_y);
%     disp(g(i));
%     disp((ll2-ll1)/jitter(i));
% end

p.method = 'LBFGS';
p.length = 100;
w = minimize_v2(w, @linear_utilities_ll, p, train_x, train_y);

% p(y=1) = Phi(x1w-x2w)
% dy/dx = phi(x1w-x2w)*w;
lm_f = (train_x(:,1:D)-train_x(:,(D+1):end))*w;
lm_dy_mu = normpdf(lm_f)*w';
inv_V = inv((train_x(:,1:D)-train_x(:,(D+1):end))'*(train_x(:,1:D)-train_x(:,(D+1):end)));
lm_dy_std = normpdf(lm_f)*diag(inv_V)'*sqrt(N);
