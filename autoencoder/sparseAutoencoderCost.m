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


num_training = 10000;
input = data(:, 1:num_training);
hidden = zeros(hiddenSize, num_training);
output = zeros(visibleSize, num_training);

W1delta = zeros(size(W1)); 
W2delta = zeros(size(W2));
b1delta = zeros(size(b1)); 
b2delta = zeros(size(b2));


%compute forward pass
for i = 1 : num_training
	hidden(:, i) = sigmoid(W1 * input (:, i) + b1);
	output(:, i) = sigmoid(W2 * hidden (:, i) + b2);	
end

%compute average activation of hidden unit rho_j
hidden_average_activation = zeros(hiddenSize);
for i = 1 : hiddenSize %iterations over all hidden units
	sum_hidden = 0;
	for j = 1 : num_training
		sum_hidden = sum_hidden + hidden(i, j);
	end
	hidden_average_activation(i) = sum_hidden/num_training;
end

%back prop
for i = 1 : num_training
	f_prime_output = zeros(visibleSize, 1);
	f_prime_output = output(:, i) .* (1 - output(:, i));
	error_output_layer = zeros(visibleSize, 1);
	error_output_layer = -(input(:, i) - output (:, i)) .* f_prime_output;

	f_prime_hidden = zeros(visibleSize, 1);
	f_prime_hidden = hidden (:, i).* (1 - hidden(:, i));
	
	%calculate error_hidden layer with respect to new sparse parameter
	error_hidden_layer = zeros(hiddenSize, 1);
	X = zeros(hiddenSize, 1);
	X = transpose(W2) * error_output_layer;
	for j = 1 : hiddenSize
        added_component = -sparsityParam/hidden_average_activation(j) + (1-sparsityParam)/(1-hidden_average_activation(j));
		adding_beta_component = beta * added_component;
		error_hidden_layer(j) = ( X(j) + adding_beta_component) * f_prime_hidden (j); 
	end

	W1delta = W1delta + error_hidden_layer * transpose (input(:, i)); 
	W2delta = W2delta + error_output_layer * transpose (hidden(:, i));
	b1delta = b1delta + error_hidden_layer;
	b2delta = b2delta + error_output_layer;

	cost_single = 1/2 * (norm(output(:, i) - input(:, i)))^2; %J(W, b, x, y)
	cost = cost + cost_single/num_training; %J(W, b) without lambda part
end

%adding weight decay components
weight_decay = 0;
for i = 1:(size(W1, 1) * size(W1, 2))
	weight_decay = weight_decay + W1(i)^2;
end
for i = 1:(size(W2, 1) * size(W2, 2))
	weight_decay = weight_decay + W2(i)^2;
end
cost = cost + (lambda/2)* weight_decay; %update lambda parameter

%adding KL divergence
KL_divergence = 0;
for i = 1 : hiddenSize
	X = sparsityParam/hidden_average_activation(i);
	Y = (1 - sparsityParam)/(1 - hidden_average_activation(i));
	KL_divergence = KL_divergence + sparsityParam * log (X) + (1 - sparsityParam) * log (Y);
end
cost = cost + (beta) * KL_divergence;

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

