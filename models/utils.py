"""
Support code that isn't the point of the exercise: data loading,
plotting, and evaluation metrics. Fully implemented so you can focus
your energy on the model internals in linear_regression.py and
logistic_regression.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_housing_data(test_size=0.2, random_state=42, standardize=True):
    """
    Loads the California Housing dataset and returns train/test splits.
    Standardizing features matters a lot for gradient descent --
    without it, features on wildly different scales (e.g. median
    income ~0-15 vs population ~100s-1000s) make convergence slow
    or unstable. Try training without standardization at some point
    to see this yourself.
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if standardize:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test, feature_names


def make_binary_target(y, threshold=None):
    """Converts the regression target into a binary above/below-median label."""
    if threshold is None:
        threshold = np.median(y)
    return (y >= threshold).astype(int)


def plot_loss_curve(loss_history, title="Training Loss"):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()


def confusion_matrix_manual(y_true, y_pred):
    """
    Returns (tp, tn, fp, fn). Implemented manually rather than
    imported, since it's worth seeing once how trivial it is.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def precision_recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_manual(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
