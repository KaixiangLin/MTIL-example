function [XX]=prox_tr(X,lambda)
[U,S,V]=svd(X,'econ');
ss=sparse(max((diag(S)-lambda),0));
tt=diag(ss); 
XX=U*(tt*V'); 
end