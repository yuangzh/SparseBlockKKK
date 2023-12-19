function [x,his,ts] = decomposition(x,Q,p,const,lambda,workingset,max_iter,accuracy,time_c)
% min_x 0.5x'Qx + p'x + lambda ||x||_0
% [random;greedy]

randn('seed',0); rand('seed',0);

n = length(p);
his = [];

ts = [];
t1 = clock();


theta = 1e-5;
grad = Q*x+p;
fobj_smooth = (grad+p)'*x/2;

last_k = 50;
fobj_old = fobj_smooth +  const + lambda * nnz(x);
changes = ones(last_k,1);

for iter = 1:max_iter
    
    B = [randperm(n,workingset(1))'; find_index(x,Q,p,lambda,workingset(2))];
    B = unique(B(:));
    
    fobj = fobj_smooth +  const + lambda * nnz(x);
%     if(~mod(iter,100))
%         fprintf('iter:%d, fobj:%f\n',iter,fobj);
%     end
    rel_change = abs((fobj - fobj_old)/max(1,fobj_old));
    changes = [changes(2:end);rel_change];
    fobj_old = fobj;
    his = [his;fobj];
    
    t2 = clock();
    tt = etime(t2,t1);
    ts = [ts;tt];
    
    H = Q(B,B);
    H1  = H + theta * eye(size(H,1));
    
    % min_{z} 0.5 (z-xk)'H1(z-xk) + <z-xk,g> + lambda ||z||_0
    x_B_old = x(B);
    x_B_new  = quad_l0r_global(H1,grad(B)-H1*x_B_old,lambda);
    x(B) = x_B_new ;
    
    grad = grad + Q(:,B)*(x_B_new-x_B_old);
    fobj_smooth = (grad+p)'*x/2;
    
%     if(mean(changes)<accuracy),break;end
  %  t2 = cputime;
    

 
    if(tt>time_c),break;end

end



function [index] = find_index(x,Q,p,lambda,num)
% find the index based on the zero-order information
% min_x 0.5x'Qx+p'x + lambda||x||_0

if(num==0)
    index = [];
    return;
end

[Z] = find(x==0);
[S] = find(x~=0);
diagQ = diag(Q);
qx = Q*x;


% fobjs1 = zeros(length(Z),1);
% for k=1:length(Z)
%     % we change zero to nonzero
%     i = Z(k);
%     alpha = -(p(i)+qx(i)) / diagQ(i);
%     fobjs1(k) = 0.5*alpha*alpha*diagQ(i) + alpha*qx(i) + alpha*p(i) + lambda;
% end

alphas = -(p(Z)+qx(Z))./diagQ(Z);
fobjs1 = 0.5.*alphas.*alphas.*diagQ(Z) + alphas.*qx(Z) + alphas.*p(Z) + lambda;

% fobjs2 = zeros(length(S),1);
% for k=1:length(S)
%     % we change nonzero to zero
%     j = S(k);
%     alpha = - x(j);
%     fobjs2(k) = 0.5*diagQ(j)*alpha^2 + qx(j)*alpha  + alpha*p(j) - lambda;
% end

alphas = -x(S);
fobjs2 = 0.5.*diagQ(S).*alphas.*alphas + qx(S).*alphas  + alphas.*p(S) - lambda;

fobjs(Z) = fobjs1;
fobjs(S) = fobjs2;
[~,ind]=sort(fobjs(:),'ascend');
index = ind(1:num);






function [x_best] = quad_l0r_global(A,b,lambda)
% This program solves the following l0 regularized problem
% min_x 0.5x'Ax + b'x + lambda ||x||_0
% A: n x n
% b: n x 1
% lambda: 1 x 1

% Note:
% (1) Global optimum can be garanteed.
% (2) This program is practical only if n <= 15.

n = length(b);
fobj_best = 00000;
x_best = zeros(n,1);

for k=1:n
    [seq] = combs([1:n],k);
    for i=1:size(seq,1)
        sel = seq(i,:);
        x = zeros(n,1);
        x(sel) = -(A(sel,sel))\b(sel);
        fobj = 0.5*x'*A*x + b'*x + lambda * nnz(x);
        if(fobj<fobj_best)
            fobj_best = fobj;
            x_best = x;
        end
    end
end


function P = combs(v,m)
% combs computes all possible combinations.
v = v(:)';
n = length(v);
if n == m
    P = v;
elseif m == 1
    P = v';
else
    P = [];
    if m < n && m > 1
        for k = 1:n-m+1
            Q = combs(v(k+1:n),m-1);
            P = [P; [v(ones(size(Q,1),1),k) Q]];
        end
    end
end
