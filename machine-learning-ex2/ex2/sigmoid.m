function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

	
g = 1./(ones(size(z)) + exp(-z)); %注意要./ 这样才是一个个除，否则1要写成向量才行

% =============================================================

end
