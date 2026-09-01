# ML Foundations — Phase 1

Building machine learning up from scratch, one phase at a time. Phase 1
covers linear and logistic regression implemented with raw numpy, no
sklearn, so the gradient descent math is fully understood before
moving to frameworks that hide it.

## Structure

- `models/linear_regression.py` — linear regression, gradients as TODOs
- `models/logistic_regression.py` — logistic regression, gradients as TODOs
- `models/utils.py` — data loading, plotting, metrics (fully implemented)
- `run_phase1.py` — trains both models and compares against sklearn baselines
- `notebooks/` — for exploratory plots and scratch work

## Deliverables

1. **Linear regression from scratch** — implement forward pass, MSE loss,
   and gradient descent in `linear_regression.py`. Validate against
   sklearn's `LinearRegression` on the same split.
2. **Logistic regression from scratch** — same idea, binary
   classification on an above/below-median housing value label.
   Validate with precision/recall and a manual confusion matrix.
3. **Gradient descent variants** — not yet scaffolded. Once 1 and 2
   pass, extend `fit()` to support batch size as a parameter, and
   compare batch vs. mini-batch vs. stochastic gradient descent on
   convergence speed.

## Setup

```bash
pip install numpy matplotlib scikit-learn
python run_phase1.py
```

## How to work through the TODOs

Each TODO in the model files describes the formula in a comment.
Derive it on paper first if it's not obvious why the gradient takes
that form — that derivation is most of the value of this phase.
