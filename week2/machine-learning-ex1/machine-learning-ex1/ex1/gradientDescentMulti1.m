function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    summa = zeros(1,size(X,2));
    temp = zeros(1,size(X,2));
    for j = 1:size(X,2)
        for i  = 1:m
            h = theta' * X(i,:)';
            summa(j) = summa(j) + ((h - y(i)).*X(i,j));
        end
        temp(j) = theta(j) - (alpha*(1/m)*summa(j));
        t1(iter,j) = temp(j);
    end
    theta = temp';
    t2(iter,:) = theta; 

% h = X*theta - y;
% for i = 1:size(X,2)
%     temp(i,1) = sum(h.*X(:,i));
% end
% theta = theta - (alpha/m)*temp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
