function tr = trace_norm(X)
% calculate trace norm of matrix X: m * n 
[U,S,V]=svd(X,'econ');
tr = sum(diag(S));
end