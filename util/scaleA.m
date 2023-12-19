function [A] = scaleA(A)
[m,n] = size(A);
seq = randperm(m*n,round(0.02*m*n));
A(seq) = A(seq)*100;