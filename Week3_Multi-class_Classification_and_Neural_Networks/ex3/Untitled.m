a = [1 10;10 4;5 6];
b = [1;2;2];
m = size(a, 1);
a = [ones(m, 1) a];
subindex = find(b == 2);
b_sub = b(subindex);
a_sub = a(subindex,:);
initial_theta = zeros(size(a_sub,2), 1);
[p, Ind]= max(a, [], 2)