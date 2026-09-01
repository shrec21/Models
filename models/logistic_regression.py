"""
Logistic Regression from scratch, using only numpy.

Same deal as linear_regression.py: derive the gradients yourself.
The sigmoid and binary cross-entropy derivatives are the two things
worth actually working out by hand before you code them.
"""

import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.001, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        # Implement sigmoid. Watch out for overflow on large
        z = np.clip(z, -500, 500)
        return 1 / (1+ np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # Linear combination, then squash with sigmoid
            linear_output = X @ self.weights + self.bias
            y_pred = self._sigmoid(linear_output)

            # Binary cross-entropy loss
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.loss_history.append(loss)

            # Gradients (same form as linear regression --
            # this is not a coincidence, work out why)
            error = y_pred - y
            dW = (1/n_samples) * X.T @ error    
            dB = (1/n_samples) * np.sum(error)

            # Parameter update
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

        return self

    def predict_proba(self, X):
        # Return sigmoid(X @ weights + bias)
        linear_output = X @ self.weights + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
