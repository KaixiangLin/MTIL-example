%% FUNCTION de_vectorization
%  devectorize vector z to matrix W and tensor Q  
%  
%% INPUT
%   z: vector (d*K+d*d*K) * 1 - vectorized W and Q.
%   d: dimension of feature vector
%   K: number of tasks.
%% OUTPUT
%   W: model: d * K.     - Weight matrix for all tasks. Each column represent
%   the weight vector for one task.
%   Q: model: d * d * K  - Store as d * d * K 3dim array. Q(:,:,t) is d * d matrix. Interaction tensor for all task. Each frontal
%   slice represent the interaction matrix for one task. 
%% Description
%   Each column of W will be concatenated to the first d * K elements of z. 
%   Each fiber of Q will be concatenated to the last d * d * K elements of z
%   in the order of first frontal slice of Q's all column vectors.

%% Code starts here
function [W,Q] = de_vectorization(z,d,K) % checked.

W = zeros(d,K);
Q = cell(3,1);

for i = 1:3
Q{i} = zeros(d,d,K);
end

W = reshape(z(1:d*K),d,K);
for i = 1:3
Q{i} = reshape(z((1+(i-1)*d)*d*K +1:(1+(i-1)*d+d)*d*K),d,d,K);
end

end 