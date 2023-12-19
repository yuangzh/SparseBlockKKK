% function [X,his] = proximal_gradient_l0c(X,A,B,k)

function [X,his] = proximal_gradient_l0c(X,A,B,k)
% This program solves the following l0 regularized problem
% % min_X 0.5 ||AX-B||_2^2, s.t. ||X||_0 <=k

[m,d] = size(A);
if(d<m)
[~,L] = laneig(A'*A,1,'AL');
else
[~,L] = laneig(A*A',1,'AL');
end
 
his = [];
for iter = 1:300,
    ERR = A*X-B;
    grad = A'*ERR;
    fobj = 0.5*norm(ERR,'fro')^2;
%     fprintf('iter:%d, fobj:%f\n',iter,fobj);
    his = [his;fobj];
    X = proj_l0(X-grad/L,k);
end