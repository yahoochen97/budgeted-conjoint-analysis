function dK = dcovPref(cov, hyp, x, z, varargin)

% dcovPref - covariance function 1st gradient for preference learning. 
% The covariance function corresponds to a prior on f(x1) - f(x2).
%
% k(x,z) = k_0(x1,z1) + k_0(x2,z2) - k_0(x1,z2) - k_0(x2,z1).
%
% dk(x,z)/dx1 = dk_0(x1,z1)/dx1 - dk_0(x1,z2)/dx1.
%
% The hyperparameters are:
%
% hyp = [ hyp_k0 ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Yehu Chen, 2023-05-02.
%
% See also COVFUNCTIONS.M.

    if nargin<3, K = strrep(feval(cov{:}),'D','D/2'); return; end     % no of params

    x1 = x(:,1:end/2); x2 = x(:,1+end/2:end);
    if nargin<4 || isempty(z), z = x; end

    z1 = z(:,1:end/2); z2 = z(:,1+end/2:end);
    % dk_0(x1,z1)/dx1
    dK =  permute(repmat(-x1,[1 1 size(z1,1)]), [1 3 2]) - permute(repmat(-z1,[1 1 size(x1,1)]),[3 1 2]);
    dK1 = feval(cov{:},hyp.cov,x1,z1,varargin{:});
    dK1 = dK.*repmat(dK1,[1 1 size(x1,2)]);
    for d=1:size(x1,2)
        dK1(:,:,d) = dK1(:,:,d)/exp(2*hyp.cov(d));
    end

    % - dk_0(x1,z2)/dx1
    dK = permute(repmat(-x1,[1 1 size(z2,1)]), [1 3 2]) - permute(repmat(-z2,[1 1 size(x1,1)]),[3 1 2]);
    dK2 = feval(cov{:},hyp.cov,x1,z2,varargin{:});
    dK2 = dK.*repmat(dK2,[1 1 size(x1,2)]);
    for d=1:size(x1,2)
        dK2(:,:,d) = dK2(:,:,d)/exp(2*hyp.cov(d));
    end

    dK = dK1 - dK2;
end
