function x = generate_x(n,k)
p = randperm(n);
x = zeros(n,1);
x(p(1:k)) = randn(k,1);