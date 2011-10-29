function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


y_negative = find(y==0);
y_positive = find(y==1);

plot(X(y_negative, 1), X(y_negative, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
plot(X(y_positive, 1), X(y_positive, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7)



% =========================================================================



hold off;

end
