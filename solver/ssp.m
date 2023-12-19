function [x_out,Rec] = ssp(K,Phi,y)
% For algorithm description, explanation and analysis, please check
% Wei Dai and Olgica Milenkovic
% "Subspace Pursuit for Compressive Sensing: Closing the
% Gap Between Performance and Complexity"%

[m,N]=size(Phi);

y_r = y;
in = 1;

cv = abs( y_r'*Phi );
[cv_sort, cv_index] = sort(cv,'descend');
cv_index = sort( cv_index(1:K) );
Phi_x = Phi(:,cv_index);
Index_save(in,:) = cv_index;

x_p = pinv(Phi_x'*Phi_x)*Phi_x' * y;
y_r = y - Phi_x*x_p;
norm_save(in) = norm(y_r);

while 1
   in = in+1;

   % find T^{\prime} and add it to \hat{T}
   cv = abs( y_r'*Phi );
   [cv_sort, cv_index] = sort(cv,'descend');
   cv_index = sort( cv_index(1:K) );
   cv_add = union(Index_save(in-1,:), cv_index);
   Phi_x = Phi(:,cv_add);

   % find the most significant K indices
   x_p = pinv(Phi_x'*Phi_x)*Phi_x' * y;
   [x_p_sort, i_sort] = sort( abs(x_p) , 'descend' );
   cv_index = cv_add( i_sort(1:K) );
   cv_index = sort( cv_index );
   Phi_x = Phi(:,cv_index);
   Index_save(in,:)=cv_index;

   % calculate the residue
   x_p = pinv(Phi_x'*Phi_x)*Phi_x' * y;
   y_r = y - Phi_x*x_p;

   norm_save(in) = norm(y_r);

   if ( norm_save(in) == 0 ) | ...
           (norm_save(in)/norm_save(in-1) >= 1)
       break;
   end
end

x_hat = zeros(N,1);
x_hat( Index_save(in,:) ) = reshape(x_p,K,1);
Rec.T = Index_save;
Rec.x_hat = x_hat;
Rec.PResidue = norm_save;

x_out = x_hat;



function X = pinv(A,tol)
%PINV   Pseudoinverse.
%   X = PINV(A) produces a matrix X of the same dimensions
%   as A' so that A*X*A = A, X*A*X = X and A*X and X*A
%   are Hermitian. The computation is based on SVD(A) and any
%   singular values less than a tolerance are treated as zero.
%
%   PINV(A,TOL) treats all singular values of A that are less than TOL as
%   zero. By default, TOL = max(size(A)) * eps(norm(A)).
%
%   Class support for input A: 
%      float: double, single
%
%   See also RANK.
 
%   Copyright 1984-2015 The MathWorks, Inc. 
A = full(A);
[U,S,V] = svd(A,'econ');
s = diag(S);
if nargin < 2 
    tol = max(size(A)) * eps(norm(s,inf));
end
r1 = sum(s > tol)+1;
V(:,r1:end) = [];
U(:,r1:end) = [];
s(r1:end) = [];
s = 1./s(:);
X = (V.*s.')*U';
