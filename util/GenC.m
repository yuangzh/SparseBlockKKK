function [seqs] = GenC(n)
for k=1:n
seqs{k} = combs([1:n],k);
end
