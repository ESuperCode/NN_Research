"""
benchmark.py

Evaluates the baseline network and the simplified boundary surrogate on an
unseen test set: accuracy, fidelity (agreement with baseline), and
per-batch inference latency at multiple batch sizes.

This script reports whatever the numbers actually are. It does not iterate
until a predetermined outcome appears -- if the surrogate loses on accuracy
or latency at some scale, that is a real result, not a bug to be tuned away.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_text_classification_data
from baseline_model import train_baseline
from boundary_simplifier import convert_to_simplified_boundary


def run_benchmark(n_components=20, batch_sizes=(1, 16, 64, 256)):
    X_train, X_test, y_train, y_test, vec, source = load_text_classification_data()
    print(f"Data source: {source}")
    print(f"Input dimensionality (TF-IDF features): {X_train.shape[1]}")
    print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}\n")

    baseline = train_baseline(X_train, y_train)
    simplified = convert_to_simplified_boundary(
        baseline, X_train, n_components=n_components
    )

    y_pred_base = baseline.predict(X_test)
    y_pred_simp = simplified.predict(X_test)

    print("=== Accuracy on held-out test set ===")
    print(f"Baseline network accuracy:   {accuracy_score(y_test, y_pred_base):.4f}"
          f"  (F1={f1_score(y_test, y_pred_base):.4f})")
    print(f"Simplified surrogate accuracy:{accuracy_score(y_test, y_pred_simp):.4f}"
          f"  (F1={f1_score(y_test, y_pred_simp):.4f})")
    print(f"Fidelity (agreement with baseline's own predictions): "
          f"{(y_pred_base == y_pred_simp).mean():.4f}\n")

    print("=== Inference latency (seconds per full batch, mean of 20 runs) ===")
    print(f"{'batch_size':>10} | {'baseline (s)':>14} | {'simplified (s)':>15} | {'speedup':>8}")
    print("-" * 58)
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    for b in batch_sizes:
        b = min(b, X_test_dense.shape[0])
        Xb = X_test_dense[:b]
        t_base = baseline.forward_pass_latency(Xb).mean()
        t_simp = simplified.forward_pass_latency(Xb).mean()
        speedup = t_base / t_simp if t_simp > 0 else float("inf")
        print(f"{b:>10} | {t_base:>14.6f} | {t_simp:>15.6f} | {speedup:>7.2f}x")

    print()
    print("Note: both models' per-call cost is ~independent of *dataset* size N")
    print("(a fixed-size forward pass and a fixed-degree polynomial both cost the")
    print("same per point regardless of how many points you evaluate). The variable")
    print("that actually stresses this comparison is INPUT DIMENSIONALITY d and the")
    print("surrogate's n_components -- try re-running with max_features and")
    print("n_components changed to see how the tradeoff shifts.")

    return baseline, simplified, (X_train, X_test, y_train, y_test, vec)


if __name__ == "__main__":
    run_benchmark()
