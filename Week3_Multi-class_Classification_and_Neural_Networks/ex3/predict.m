function prediction = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%act_layer1 = [ones(m,1) X] * (Theta1'); %output from the input layer, which is the first layer

act_layer2 = sigmoid([ones(m,1) X] * (Theta1')); %output from the first layer of hidden layer, which is the second layer
act_layer3 = sigmoid([ones(size(act_layer2, 1), 1) act_layer2]* (Theta2'));

[~, Index] = max(act_layer3, [], 2);
    if Index == 10 
        Index = 0;
    end
prediction = Index;


% =========================================================================


end
