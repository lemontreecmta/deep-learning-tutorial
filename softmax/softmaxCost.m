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

m = theta' * data ;
m = bsxfun(@minus, m, max(m, [], 1));
m = bsxfun(@rdivide, m, sum(m));
p_numerator = exp(1) .^ m;
p_denominator = 1./ sum(p_numerator, 1); % vector of length inputSize, which adds sum of all numerator for one training example
p_denominator = repmat(p_denominator, 1, numClassses); %expand it using repmat
p_matrix = p_numerator ./ p_denominator;
log_matrix = log(p_matrix);

%assume the true value matrix is already computed in groundTruth
cost = -1/ inputSize * sum(sum(groundTruth * log_matrix, 1)) + lambda/2 * (sum(sum (theta .* theta) ) );

%compute thetagrad
%for the ith example: x(:, i) * (groundTruth(:, i) - p_matrix(:, i)

thetagrad = -1/ inputSize * sum (x * (groundTruth - p_matrix), 1) + lambda * theta;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
