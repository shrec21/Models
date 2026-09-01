"""
Linear Regression from scratch, using only numpy.

The point of this file is to implement gradient descent yourself.
Every TODO below is a place where you write the actual math -- don't
look up a reference implementation, derive it. If you get stuck, work
the derivative out on paper first, then translate it into numpy.
"""

import numpy as np


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=200):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize parameters. Zeros are fine for linear regression
        # (no symmetry-breaking issue like in neural nets).
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # TODO 1: Compute predictions
            y_pred = X @ self.weights + self.bias

            # TODO 2: Compute the MSE loss for this iteration
            loss = (1/n_samples) * sum((y_pred - y)**2)
            
            self.loss_history.append(loss)

            # Compute gradients w.r.t. weights and bias
            dW = (2/n_samples) * X.T @ (y_pred - y)
            dB = (2/n_samples) * sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        """R^2 score, so you can compare against sklearn's baseline."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
