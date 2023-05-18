function d2K = d2covPref(cov, hyp, x, z, varargin)

% dcovPref - covariance function 2nd gradient for preference learning. 
% The covariance function corresponds to a prior on f(x1) - f(x2).
%
% k(x,z) = k_0(x1,z1) + k_0(x2,z2) - k_0(x1,z2) - k_0(x2,z1).
% d2k(x,z)/dx1/dz1 = d2k_0(x1,z1)/dx1/dz1.
%
% we are only interested in d2k(x,z)/dx1/dz1 at x=z.
%
% The hyperparameters are:
%
% hyp = [ hyp_k0 ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Yehu Chen, 2023-05-03.
%
% See also COVFUNCTIONS.M.

    if nargin<3, K = strrep(feval(cov{:}),'D','D/2'); return; end     % no of params

    x1 = x(:,1:end/2); 
    if nargin<4 || isempty(z), z = x; end
    z1 = z(:,1:end/2);
    n = size(x1,1); D=size(x1,2);
    
    % d2k_0(x1,z1)/dx1/dz1 at x1=z1
    K = feval(cov{:},hyp.cov,x1,z1,varargin{:});
    d2K = zeros(n,D,D);
    for d=1:D
        d2K(:,d,d) = diag(K)./exp(2*hyp.cov(d));
    end

end
