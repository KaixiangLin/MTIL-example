%% FUNCTION vectorization
%  vectorize matrix W and tensor Q into a vector. 
%  
%% INPUT
%   W: model: d * K.     - Weight matrix for all tasks. Each column represent
%   the weight vector for one task.
%   Q: model: d * d * K  - Store as d * d * K 3dim array. Q(:,:,t) is d * d matrix. Interaction tensor for all task. Each frontal
%   slice represent the interaction matrix for one task. 

%% OUTPUT
%   z: vector (d*K+d*d*K) * 1 - vectorized W and Q.
%
%% Description
%   Each column of W will be concatenated to the first d * K elements of z. 
%   Each fiber of Q will be concatenated to the last d * d * K elements of z
%   in the order of first frontal slice of Q's all column vectors.

%% Code starts here
function z = vectorization(W, Q) % checked.

[d, K] = size(W);

z = zeros((1+d*K)*d,1);

dK = d*K;
N  = dK*d;
z(1:dK) = reshape(W,dK,1);
for i = 1:3
    Qt = Q{i};
    z(dK+1+(i-1)*N:dK+i*N) = reshape(Qt,N,1);
end


end 