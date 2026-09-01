"""
Entry point for Phase 1. Run this once you've filled in the TODOs
in models/linear_regression.py and models/logistic_regression.py.

    python run_phase1.py
"""

from sklearn.linear_model import LinearRegression, LogisticRegression

from models.linear_regression import LinearRegressionScratch
from models.logistic_regression import LogisticRegressionScratch
from models.utils import (
    load_housing_data,
    make_binary_target,
    plot_loss_curve,
    precision_recall,
)


def run_linear_regression():
    print("\n--- Linear Regression ---")
    X_train, X_test, y_train, y_test, feature_names = load_housing_data()

    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    plot_loss_curve(model.loss_history, title="Linear Regression Loss")

    r2_scratch = model.score(X_test, y_test)
    print(f"Scratch model R^2: {r2_scratch:.4f}")

    baseline = LinearRegression().fit(X_train, y_train)
    r2_sklearn = baseline.score(X_test, y_test)
    print(f"sklearn baseline R^2: {r2_sklearn:.4f}")
    print("These should be close. If they're not, your gradients are likely wrong.")


def run_logistic_regression():
    print("\n--- Logistic Regression ---")
    X_train, X_test, y_train, y_test, feature_names = load_housing_data()
    y_train_bin = make_binary_target(y_train)
    y_test_bin = make_binary_target(y_test, threshold=None)

    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train_bin)
    plot_loss_curve(model.loss_history, title="Logistic Regression Loss")

    preds = model.predict(X_test)
    accuracy = (preds == y_test_bin).mean()
    precision, recall = precision_recall(y_test_bin, preds)
    print(f"Scratch model accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train_bin)
    print(f"sklearn baseline accuracy: {baseline.score(X_test, y_test_bin):.4f}")


if __name__ == "__main__":
    run_linear_regression()
    run_logistic_regression()
