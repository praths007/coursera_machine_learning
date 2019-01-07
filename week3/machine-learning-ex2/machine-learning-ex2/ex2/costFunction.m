function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

sigm = @(z) (1 ./ (1 .+ e.^(-z)));

# thetaX is 1 / 1 - e^-thetaX
% cost = 1 / m(-yT log(h(thetaX)) - (1-y)T log(1 - h(thetaX))
J = (1 / m) * (-y' * log(sigm(X * theta)) - (1 - y)' * log(1 - sigm(X * theta)));

% gradient = partial derivative of cost function which is
% grad = 1/m(XT(g(Xtheta) - y)) where g(Xtheta) equals 1 / 1 - e^-thetaX or sigmoid
grad = (1 / m) * (X' * (sigm(X * theta) - y));





% =============================================================

end
