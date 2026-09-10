"""
Linear Regression from scratch, using only numpy.

The point of this file is to implement gradient descent yourself.
Every TODO below is a place where you write the actual math -- don't
look up a reference implementation, derive it. If you get stuck, work
the derivative out on paper first, then translate it into numpy.
"""

import numpy as np


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=None):
        """
        batch_size controls which gradient descent variant you're running:
          - batch_size=None (or equal to n_samples) -> full batch gradient descent
          - batch_size=1                            -> stochastic gradient descent (SGD)
          - anything in between                     -> mini-batch gradient descent
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,)
        """
        n_samples, n_features = X.shape
        batch_size = self.batch_size if self.batch_size is not None else n_samples

        # Initialize parameters. Zeros are fine for linear regression
        # (no symmetry-breaking issue like in neural nets).
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # TODO (Phase 1, deliverable 3): Randomly sample `batch_size` rows
            # from X and y to use for this iteration's gradient computation.
            # Hint: np.random.choice(n_samples, size=batch_size, replace=False)
            # gives you random row indices; use them to index into X and y,
            # e.g. X_batch = X[indices], y_batch = y[indices].
            # If batch_size == n_samples, this should behave identically to
            # your original full-batch implementation.

            indices = np.random.choice(n_samples, size=batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            # Forward pass -- same as before, just on X_batch/y_batch now.
            y_pred = X_batch @ self.weights + self.bias


            # Loss on the full dataset (not just the batch), so the loss
            # curve is a fair comparison across batch sizes. This does NOT
            # affect the gradient step below, it's purely for tracking.
            full_pred = X @ self.weights + self.bias
            loss = np.mean((full_pred - y) ** 2)
            self.loss_history.append(loss)

            # Gradients computed from the batch only.
            error = y_pred - y_batch
            dW = (2 / batch_size) * (X_batch.T @ error)
            dB = (2 / batch_size) * np.sum(error)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

        return self

    def predict(self, X):
        # TODO 5: Return X @ weights + bias
        return X @ self.weights + self.bias

    def score(self, X, y):
        """R^2 score, so you can compare against sklearn's baseline."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)