function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
h = 0;
alpha = 0.01;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h = X*theta;
    sum_theta = zeros(size(theta, 1), 1);
    for j = 1:size(X,2)
      for i = 1:m
        sum_theta(j,1) = sum_theta(j,1) + (h(i,1) - y(i,1))*(X(i,j));
      endfor
    endfor
    for i = 1:size(theta,1)
      theta(i,1) = theta(i,1) - alpha*(1/m)*sum_theta(i,1);
    endfor

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
