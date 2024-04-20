function K = covSEardLINiso(hyp, x, z, i)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure, plus Linear cov. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P1)*(x^p - x^q)/2) + x^p'*inv(P2)*x^q
%
% where the P1 matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. 
% The P2 matrix is ell^2 times the unit matrix. The hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) 
%         log(ell) ]
%
% Note that there is no bias term; use covConst to add a bias.
%
% Copyright (c) by Yehu Chen, 2023-04-20.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '(D+2)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

[n,D] = size(x);
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance
ell2 = exp(hyp(D+2));                                  % linear iso length scale

% precompute squared distances
if dg                                                               % vector kxx
  K1 = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K1 = sq_dist(diag(1./ell)*x');
  else                                                   % cross covariances Kxz
    K1 = sq_dist(diag(1./ell)*x',diag(1./ell)*z');
  end
end

x2 = x/ell2;
% precompute inner products
if dg                                                               % vector kxx
  K2 = sum(x2.*x2,2);
else
  if xeqz                                                 % symmetric matrix Kxx
    K2 = x2*x2';
  else                                                   % cross covariances Kxz
    z2 = z/ell2;
    K2 = x2*z2';
  end
end

K = sf2*exp(-K1/2) + K2;                                             % covariance
if nargin>3                                                        % derivatives
  if i<=D                                              % length scale parameters
    if dg
      K = K1*0;
    else
      if xeqz
        K = K1.*sq_dist(x(:,i)'/ell(i));
      else
        K = K1.*sq_dist(x(:,i)'/ell(i),z(:,i)'/ell(i));
      end
    end
  elseif i==D+1                                            % magnitude parameter
    K = 2*K1;
  elseif i==D+2
    if dg
      K = -2*sum(x2.*x2,2);
    else
      if xeqz
        K = -2*(x2*x2');
      else
        K = -2*x2*z2';
      end
    end
  else
    error('Unknown hyperparameter')
  end
end

