function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);%number of theta
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J_matrix_unregularized = (X * theta - y);
J_unreg = (1/(2*m)) * (ones(1,m) * (J_matrix_unregularized.^2));
J_regularizeditem = (lambda/(2*m)) * ((ones(1,n - 1) * (theta(2:n) .^ 2)));
J = J_unreg + J_regularizeditem;

grad = (1/m) * ((X') * (X * theta - y)) + (lambda/m) * theta;
grad(1) = (1/m) * ((X(:,1)') * (X * theta - y));% no regularization taken on grad(0)

% =========================================================================

grad = grad(:);

end
