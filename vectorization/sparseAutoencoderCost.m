function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

num_training = 10;
input = data(:, 1:num_training);
hidden = zeros(hiddenSize, num_training);
output = zeros(visibleSize, num_training);

W1delta = zeros(size(W1)); 
W2delta = zeros(size(W2));
b1delta = zeros(size(b1)); 
b2delta = zeros(size(b2));

%use vectorization to compute forward pass
hidden = sigmoid(W1 * input + repmat(b1, 1, num_training));
output = sigmoid(W2 * hidden + repmat(b2, 1, num_training));

%use vectorization to compute average activation of hidden unit rho_j
hidden_average_activation = zeros(hiddenSize);
hidden_average_activation = mean(hidden, 2);


%back prop
sparsityDelta = -sparsityParam ./ hidden_average_activation + (1-sparsityParam) ./ (1-hidden_average_activation);
error_output_layer = -(input - output) .* fprime(output);
error_hidden_layer = (W2' * error_output_layer + beta * repmat(sparsityDelta, 1, num_training)).* fprime(hidden);
W1delta += error_hidden_layer * input'; 
W2delta += error_output_layer * hidden';
b1delta += sum(error_hidden_layer, 2);
b2delta += sum(error_output_layer, 2);		

%update cost function, weight decay, KL divergence
cost_square_error = 1/ (2 * num_training) * sum( sum ( (output - input) .^2 ));
weight_decay = (lambda/2) * (sum(sum(W1 .^2)) + sum (sum (W2 .^2)) );
log_1 = sparsityParam * log (sparsityParam ./ hidden_average_activation);
log_2 = (1 - sparsityParam) * log ( (1 - sparsityParam) ./ (1 - hidden_average_activation) );
KL = beta * (sum(log_1) + sum(log_2));

cost += cost_square_error + weight_decay + KL;

%update Wgrad, bgrad
W1grad = (1/num_training) * W1delta + lambda * W1;
W2grad = (1/num_training) * W2delta + lambda * W2;
b1grad = (1/num_training) * b1delta;
b2grad = (1/num_training) * b2delta;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function f = fprime(x)
	
	f = x .* (1 - x);

end