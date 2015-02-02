function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

%create matrix with 0 and 1 (1: numCase = 1:10)
groundTruth = full(sparse(labels, 1:numCases, 1)); 
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

%compute cost

x = data;
t = theta * x;
%t = bsxfun(@minus, t, max(t, [], 1));
%t = bsxfun(@rdivide, t, sum(t));
p_numerator = exp(1) .^ t;
denominator = sum(p_numerator, 1); % vector of length numCases, which adds sum of all numerator for one training example
p_denominator = repmat(denominator, 1, numClasses); %expand it using repmat
%repmat and reshape should be used carefully. p_denominator is the replication
%of sum over all values in one instance) 
p_denominator = reshape(p_denominator, numCases, numClasses );
p_denominator = p_denominator'; 
p_matrix = p_numerator ./ p_denominator;
log_matrix = log(p_matrix);

%assume the true value matrix is already computed in groundTruth

cost = ( -1/ numCases ) * sum(sum(groundTruth .* log_matrix)) + (lambda/2) * (sum(sum (theta .* theta) ) );

%compute thetagrad
%for the ith example: x(:, i) * (groundTruth(:, i) - p_matrix(:, i)
grad = zeros(size(thetagrad));

%to use vectorization and completely eliminate for loop seems to involve use of 3D
%matrix, which I do not implement 

for j = 1:numClasses

    Z = zeros(size(x));
    S = zeros(numCases);
    S = groundTruth(j,:) - p_matrix(j,:);
    S = repmat(S, 1, inputSize);
    S = reshape(S, numCases, inputSize);
    Z = x .* S';
  
    Z_sum = sum (Z, 2)';    
    thetagrad(j, :) = ((- 1/ numCases) * Z_sum) + lambda * theta(j,:);
end
   
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
