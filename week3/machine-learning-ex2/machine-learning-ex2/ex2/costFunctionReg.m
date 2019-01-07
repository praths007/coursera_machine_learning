function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

sigm = @(z) (1 ./ (1 .+ e.^(-z)));

# thetaX is 1 / 1 - e^-thetaX
% cost = 1 / m(-yT log(h(thetaX)) - (1-y)T log(1 - h(thetaX))
J = (1 / m) * (-y' * log(sigm(X * theta)) - ...
                    (1 - y)' * log(1 - sigm(X * theta)) + ...
                          ((lambda / 2) * sum(theta(2:end) .^2))); % regularization 
                                                                % parameter

% gradient = partial derivative of cost function which is
% grad = 1/m(XT(g(Xtheta) - y)) where g(Xtheta) equals 1 / 1 - e^-thetaX or sigmoid
% + regularization term
% theta(0) / constant / intercept will not have regularization
% both will be separetely calculated i.e. first J(theta0) and then J(thetaj)
% where j = 1 to n (n being number of features)

grad0 = (1 / m) * ((X(:,1)' * (sigm(X * theta) - y)));


gradj = (1 / m) * ((X(:,2:end)' * (sigm(X * theta) - y))) + ((lambda / m) * theta(2:end));

grad = [grad0; gradj];




% =============================================================

end
