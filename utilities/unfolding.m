%% FUNCTION vectorization
%  Mold k unfolding of K- way tensor Q. 
%  
%% INPUT
%   Q: tensor: d1 * d2 *... * dK  - Store as K dim array. 
%   k: integer range from [1,K]
%% OUTPUT
%   Qt: matrix: dk * (\prod_{i=1}^K (i!=k) di).
%
%% Description
%  Concatnate mode k fiber of tensor Q to a matrix Qt in order k, k+1,...nd
%  1,2,...,k-1.
%
%% Code starts here


function Qt = unfolding(Q,k)

 
nd = ndims(Q);

if k ~=1
    Q = permute(Q,[k:nd,1:k-1]);
end

Qt = Q(:,:);

end