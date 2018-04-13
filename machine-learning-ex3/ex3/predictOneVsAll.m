function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       




[c,i] = max(sigmoid(X * all_theta'), [], 2);
p = i;  

%我们正是通过Matlab 的 max 函数，求得矩阵sigmoid(X * all_theta') 的每一行的最大值(0到1之间，数值越趋于1，则认为概率越大)。将每一行的中的最大值 的索引 赋值给向量i。其中，sigmoid(X * all_theta') 是一个5000行乘10列的矩阵 

%向量p就是预测的结果向量。而由于 sigmoid(X * all_theta') 有10列，故 p 中每个元素的取值范围为[1,10]，即分别代表了数字 0-9（其中10 表示 0）


% =========================================================================


end
