function W_new = least_method_cell(X, Y, W, lambda)

[d,K] = size(W);
I = eye(d);
W_new = zeros(d,K);

for i = 1:K
    yt = Y{i};
    Xt = X{i};
    W_new(:,i) = (Xt'*Xt+ lambda*I)\Xt'*yt;
end


end