"""
main.py

Runs the full pipeline end to end:
  1. Load text classification data (real 20 Newsgroups if you have network,
     synthetic offline fallback otherwise -- see data_utils.py).
  2. Train the baseline neural network.
  3. Autotune the simplified closed-form surrogate: search a small grid of
     configs and pick the FASTEST one whose accuracy stays within
     --max-accuracy-drop of baseline (see boundary_simplifier.py).
  4. Benchmark accuracy, fidelity, and latency on unseen test data.
  5. Build the interactive 3D visualization (open the resulting .html file
     in a browser; needs internet access to load the Three.js library from
     its CDN the first time you open it).

Usage:
    python main.py
    python main.py --max-features 3000 --max-accuracy-drop 0.001 --grid-res 32
"""

import argparse
from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_text_classification_data
from baseline_model import train_baseline
from boundary_simplifier import autotune_surrogate
from visualize_3d import build_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=3000,
                         help="TF-IDF vocabulary size (input dimensionality).")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.001,
                         help="Max allowed accuracy drop vs. baseline, as a fraction "
                              "(0.001 = 0.1 percentage points). The autotuner picks "
                              "the fastest surrogate config meeting this bar.")
    parser.add_argument("--grid-res", type=int, default=28,
                         help="Grid resolution per axis for the 3D surface extraction.")
    parser.add_argument("--out", type=str,
                         default="/mnt/user-data/outputs/boundary_visualization.html")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, vec, source = load_text_classification_data(
        max_features=args.max_features
    )
    print(f"Data source: {source}")
    print(f"Input dimensionality: {X_train.shape[1]}  "
          f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}\n")

    baseline = train_baseline(X_train, y_train)
    simplified, tuning_report = autotune_surrogate(
        baseline, X_train, y_train, max_accuracy_drop=args.max_accuracy_drop
    )
    print(f"\nChosen surrogate config: {tuning_report['chosen_config']}  "
          f"[{tuning_report['status']}]\n")

    y_pred_base = baseline.predict(X_test)
    y_pred_simp = simplified.predict(X_test)
    print("=== Accuracy ===")
    print(f"Baseline:   {accuracy_score(y_test, y_pred_base):.4f} "
          f"(F1={f1_score(y_test, y_pred_base):.4f})")
    print(f"Simplified: {accuracy_score(y_test, y_pred_simp):.4f} "
          f"(F1={f1_score(y_test, y_pred_simp):.4f})")
    print(f"Fidelity:   {(y_pred_base == y_pred_simp).mean():.4f}\n")

    t_base = baseline.forward_pass_latency(
        X_test.toarray() if hasattr(X_test, "toarray") else X_test
    ).mean()
    t_simp = simplified.forward_pass_latency(
        X_test.toarray() if hasattr(X_test, "toarray") else X_test
    ).mean()
    print("=== Latency (full test batch) ===")
    print(f"Baseline:   {t_base:.6f}s")
    print(f"Simplified: {t_simp:.6f}s")
    print(f"Speedup:    {t_base / t_simp:.2f}x "
          f"({'simplified faster' if t_simp < t_base else 'baseline faster'})\n")

    build_visualization(
        baseline, simplified, X_train, y_train, X_test, y_test,
        out_path=args.out, grid_res=args.grid_res
    )


if __name__ == "__main__":
    main()
