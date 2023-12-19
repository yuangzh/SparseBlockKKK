function [x_best,his,ts]=AccerlatedProximalGradient_Constant_Stepsize(x,HandleObjSmooth,HandleObjNonSmooth,HandleProx,L,maxiter,accuracy,time_c)
% This program solves the following optimization problem:
% f(x) + g(x)
% where we assume that f is smooth g is non-smooth
% HandleObjSmooth:           x   ->  [fobj,grad]
% HandleObjNonSmooth:        x   ->  [fobj]
% HandleProx:          [theta,a] ->  arg min_{x} 0.5 theta || x - a ||^2 + g(x)

% One xample:

% function example_LeastR
% clear, clc;
% %  min  1/2 || A x - y||^2 + lambda * ||x||_1
% % f(x) + g(x)
% % 0.5 L ||x-xt||^2 + <x-xt,g> + g(x)
% % 0.5 L ||x-(xt-g/L)||^2 + g(x)
% % proximal mapping:
% 
% % 0.5 theta ||x - a||^2 + g(x)
% 
% 
% m=1000;  n=100;    % The data matrix is of size m x n
% A=randn(m,n);       % the data matrix
% y = randn(m,1);
% lambda=0.2;
% HandleObjSmooth = @(x)computeObj(x,A,y);
% HandleObjNonSmooth = @(x)lambda*sum(abs(x));
% x=zeros(n,1);
% HandleProx = @(theta,a)computeprox(theta,a,lambda);
% [x1, his]= AccerlatedProximalGradient(x,HandleObjSmooth,HandleObjNonSmooth,HandleProx);
% plot(his)
% 
% function [fobj,grad] = computeObj(x,A,y)
% diff = A*x-y;
% fobj = 1/2*norm(diff)^2 ;
% grad = A'*diff ;
% 
% function [x] = computeprox(theta,a,lambda)
% % 0.5 theta ||x - a||^2 + g(x)
% [x] = threadholding_l1(a,lambda/theta);
% 
% function [x] = threadholding_l1(a,lambda)
% % solving the following OP:
% % min_{x} 0.5 ||x - a||^2 + lambda * sum(abs(x))
% x = sign(a).*max(0,abs(a)-lambda);

% last modified: 2016-01-29

[n,d]=size(x);
flag=0;
xp=x;
xxp=zeros(n,d);
alpha=1; s=x;
x_best = x;
t1 = clock();
ts = [];
last_k = 50;
fobj_old = HandleObjSmooth(x)+ HandleObjNonSmooth(x);
changes = ones(last_k,1);

for iterStep=1:maxiter
    [fobj_small,g_old] = HandleObjSmooth(s);
    fobj = fobj_small + HandleObjNonSmooth(s);
    his(iterStep) = fobj;
    
        rel_change = abs((fobj - fobj_old)/max(1,fobj_old));
    changes = [changes(2:end);rel_change];
    fobj_old = fobj;
    
    
    t2 = clock();
    tt = etime(t2,t1);
    ts = [ts;tt];
    xp=x;
    % min_{x} 0.5 L ||x-(xt-g/L)||^2 + z * ||x||_1
    [x]=HandleProx(L,s-g_old/L);
    
    
    alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;
    beta=(alphap-1)/alpha;    s=x + beta* xxp;
    xxp=x-xp;
%       if(mean(changes)<accuracy),break;end
%     if (iterStep>50 && flag),break;end
  if(tt>time_c),break;end
end
his = his(:);

function [r] = mdot(x,y)
r = sum(sum(x.*y));