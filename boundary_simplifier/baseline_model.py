"""
baseline_model.py

Baseline neural network: a small MLP with a single sigmoid output node,
trained on TF-IDF text features. Threshold at 0.5 -> class 1, else class 0.

Uses sklearn's MLPClassifier rather than a hand-rolled PyTorch/TF model so
this runs anywhere without extra heavy dependencies. Swap in a PyTorch model
if you want -- the rest of the pipeline only needs an object exposing
`.predict_proba(X) -> P(class=1)` and works against dense or sparse X.
"""

import time
import numpy as np
from sklearn.neural_network import MLPClassifier


class BaselineTextClassifier:
    """Thin wrapper so downstream code has a stable, simple interface
    regardless of the underlying library (sklearn MLP, PyTorch, etc.)."""

    def __init__(self, hidden_layer_sizes=(64, 32), random_state=42, max_iter=400):
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        """Returns P(class=1), i.e. the sigmoid output of the single output node."""
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def forward_pass_latency(self, X, n_repeats=20):
        """Measures per-call wall-clock latency of the full network forward pass."""
        # warm-up
        self.predict_proba(X[:1])
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            self.predict_proba(X)
            times.append(time.perf_counter() - t0)
        return np.array(times)


def train_baseline(X_train, y_train):
    model = BaselineTextClassifier()
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    from data_utils import load_text_classification_data
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test, vec, source = load_text_classification_data()
    model = train_baseline(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Baseline test accuracy ({source}): {acc:.4f}")
