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

summation = 0;
for i=1:m
	firstPart = -y(i,1)*(log(sigmoid(X(i,1:size(X,2))*theta)));
	secondPart = -(1-y(i,1))*(log(1-(sigmoid(X(i,1:size(X,2))*theta))));
	tempSum = firstPart + secondPart;
	summation = summation + tempSum;
endfor

summationTheta = 0;
for i=2:size(theta,1)
	summationTheta = summationTheta + (theta(i,1))**2;
endfor

J = (summation/m) + (lambda*(summationTheta/(2*m)));

for j=1:size(theta,1) 
	sums = 0;
	for i=1:m
		sums = sums + ((sigmoid(X(i,1:size(X,2))*theta) - y(i))*X(i,j));
	endfor
	if j>1
		grad(j,1) = (sums + lambda*theta(j,1))/m;
	else
		grad(j,1) = (sums/m);
	end
endfor

% =============================================================

end
