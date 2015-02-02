function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.


%predict file
%need to include the new p_denominator computation

m = theta' * data ;
m = bsxfun(@minus, m, max(m, [], 1));
m = bsxfun(@rdivide, m, sum(m));
p_numerator = exp(1) .^ m;
p_denominator = 1./ sum(p_numerator, 1); 
p_denominator = repmat(p_denominator, 1, numClassses); 
p_matrix = p_numerator ./ p_denominator;
(M, I) = max(p_matrix, [], 1);
pred = I;





% ---------------------------------------------------------------------

end


