"""
Deliverable 3: compare batch, mini-batch, and stochastic gradient descent
on the same linear regression problem.

Fill in the TODO in models/linear_regression.py first, then run:

    python compare_gd_variants.py
"""

import matplotlib.pyplot as plt

from models.linear_regression import LinearRegressionScratch
from models.utils import load_housing_data


def main():
    X_train, X_test, y_train, y_test, _ = load_housing_data()
    n_samples = X_train.shape[0]

    # Each variant gets its own learning rate. Smaller batches produce
    # noisier, higher-variance gradients, so they need a smaller learning
    # rate to stay stable -- otherwise one unlucky sample can throw the
    # weights way off, which is exactly what caused the spikes and the
    # negative R^2 you just saw.
    variants = {
        "Batch (full dataset)": (n_samples, 0.1),
        "Mini-batch (size=64)": (64, 0.05),
        "Stochastic (size=1)": (1, 0.001),
    }

    plt.figure(figsize=(8, 5))

    for label, (batch_size, learning_rate) in variants.items():
        model = LinearRegressionScratch(
            learning_rate=learning_rate, n_iterations=500, batch_size=batch_size
        )
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        print(f"{label} (lr={learning_rate}): final R^2 = {r2:.4f}")
        plt.plot(model.loss_history, label=label, alpha=0.8)

    plt.xlabel("Iteration")
    plt.ylabel("Loss (full dataset, log scale)")
    plt.yscale("log")
    plt.title("Batch vs Mini-batch vs Stochastic Gradient Descent")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("gd_variants_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()