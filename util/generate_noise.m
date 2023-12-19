function [o] = generate_noise(n,type)
 
switch type
    case 1
        o = 10*randn(n,1);
    case 2
        o = 10*randn(n,1);
        seq = randperm(n,round(0.02*n));
        o(seq) = o(seq)* 100;
end


