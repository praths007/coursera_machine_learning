function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% adding bias values 1 to vector X
a1 = [ones(size(X, 1), 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% computing cost without vectorization for time being
% will be vectorized later if time permits
% error part (w/o regularization)

individual_theta_k = zeros(size(num_labels));
individual_train = zeros(size(m));

for i = 1:m
  for k = 1:num_labels
    individual_theta_k(k) = (-(y(i) == k) * log(a3(i,k)) - ...
                            (1 - (y(i) == k)) * log(1 - a3(i,k)));

  endfor;
  individual_train(i) = sum(individual_theta_k);
endfor;
  
J = (1 / m) * sum(individual_train);
  
  
## regularized cost
  
## considering from column 2 onwards as
## regularization paramter for bias (1st column should not be considered
## so i goes starts from 2 instead of 1  
## i is column and j is row (according to formula in lecture pdf)
theta1 = 0;
for i = 2 : size(Theta1, 2)
  for j = 1 : size(Theta1, 1)
    theta1 = theta1 + Theta1(j, i) .^2;
  endfor;
endfor;


theta2 = 0;
for i = 2 : size(Theta2, 2)
  for j = 1 : size(Theta2, 1)
    theta2 = theta2 + Theta2(j, i) .^2;
  endfor;
endfor;
  
reg = (lambda / (2 * m)) * (theta1 + theta2); 
  
J = J + reg;
  
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


delta3 = zeros(size(m), size(num_labels));

for i = 1:m
  for k = 1:num_labels
      delta3(i,k) = a3(i,k) - (y(i) == k);
  endfor;
endfor;

% implemented according to backprop intuition lecture
% using the same formula (which uses vectorization)
delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Theta1_grad = (1 / m) * (delta2' * a1);
Theta2_grad = (1 / m) * (delta3' * a2);


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


theta_reg1 = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
theta_reg2 = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad + theta_reg1;
Theta2_grad = Theta2_grad + theta_reg2;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
