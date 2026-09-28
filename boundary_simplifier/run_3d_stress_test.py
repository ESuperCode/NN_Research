"""
run_3d_stress_test.py

Runs the full pipeline on genuinely hard, natively-3D two-class patterns
(helix, swiss_roll, xor, spheres) instead of the text-classification data.
Because the input is already 3D, the visualization needs no PCA projection
-- what you see is the EXACT decision surface of both the baseline network
and the simplified surrogate, side by side, with the real data points.

This is the honest stress test: can a closed-form surrogate actually
capture a non-convex / wrap-around / disconnected boundary? Try both
`--basis poly` and `--basis rbf` on `--pattern helix` or `--pattern xor`
and compare -- a low-degree polynomial should visibly fail to wrap around
the helix or separate disjoint octants, while the RBF/Fourier-feature
surrogate should do noticeably better, at some extra evaluation cost.

Usage:
    python run_3d_stress_test.py --pattern helix --basis rbf
    python run_3d_stress_test.py --pattern xor --basis poly --poly-degree 4
    python run_3d_stress_test.py --pattern spheres --basis rbf --n-rbf-features 300
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from synthetic_3d_datasets import load_3d_dataset
from baseline_model import BaselineTextClassifier  # generic MLP wrapper, works for any input
from boundary_simplifier import convert_to_simplified_boundary, autotune_surrogate
from visualize_3d import build_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=["helix", "swiss_roll", "xor", "spheres"],
                         default="helix")
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[200, 100, 50],
                         help="Baseline MLP hidden layer sizes. These patterns need "
                              "real capacity to fit well -- e.g. helix needs roughly "
                              "this size or larger to clear ~95% accuracy; smaller "
                              "networks under-fit the pattern itself, which would "
                              "confound 'the surrogate failed' with 'the baseline "
                              "never learned it either'.")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--basis", choices=["poly", "rbf"], default="rbf",
                         help="Closed-form surrogate family (ignored if --autotune is "
                              "set). 'poly' is a low-degree polynomial (will struggle "
                              "on wrap-around/disjoint boundaries). 'rbf' is a random-"
                              "Fourier-feature surrogate (handles them far better).")
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--n-rbf-features", type=int, default=250)
    parser.add_argument("--rbf-gamma", type=float, default=0.3)
    parser.add_argument("--autotune", action="store_true",
                         help="Ignore --basis/--poly-degree/--n-rbf-features and instead "
                              "search for the fastest surrogate meeting --max-accuracy-drop "
                              "(same search compare_models.py uses). Use this when you just "
                              "want speed and don't care which basis wins; use manual mode "
                              "(the default) to deliberately compare bases, as this script "
                              "was originally built for.")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.001,
                         help="Only used with --autotune. Fraction (0.001 = 0.1pp).")
    parser.add_argument("--grid-res", type=int, default=45,
                         help="Higher than the text pipeline default -- these "
                              "boundaries are more intricate and need finer sampling.")
    parser.add_argument("--out", type=str,
                         default="/mnt/user-data/outputs/stress_test_visualization.html")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, source = load_3d_dataset(
        args.pattern, n_samples=args.n_samples
    )
    print(f"Data source: {source}")
    print(f"Train: {X_train.shape}  Test: {X_test.shape}\n")

    baseline = BaselineTextClassifier(hidden_layer_sizes=tuple(args.hidden_layers),
                                       max_iter=args.max_iter)
    baseline.fit(X_train, y_train)
    base_train_acc = accuracy_score(y_train, baseline.predict(X_train))
    if base_train_acc < 0.9:
        print(f"[warning] baseline network only reached {base_train_acc:.2f} TRAIN "
              f"accuracy on this pattern -- it hasn't learned the shape well itself. "
              f"Increase --hidden-layers / --max-iter before judging the surrogate.\n")

    if args.autotune:
        simplified, tuning_report = autotune_surrogate(
            baseline, X_train, y_train, max_accuracy_drop=args.max_accuracy_drop,
            skip_pca=True, clip_non_negative=False,
        )
        print(f"Chosen surrogate config: {tuning_report['chosen_config']}  "
              f"[{tuning_report['status']}]\n")
    else:
        simplified = convert_to_simplified_boundary(
            baseline, X_train,
            skip_pca=True,              # already 3D -- no reduction needed
            clip_non_negative=False,    # real-valued coordinates, can be negative
            basis=args.basis,
            poly_degree=args.poly_degree,
            n_rbf_features=args.n_rbf_features,
            rbf_gamma=args.rbf_gamma,
            n_synthetic_per_point=6,
            jitter_std=0.4,
        )

    y_pred_base = baseline.predict(X_test)
    y_pred_simp = simplified.predict(X_test)

    print("=== Accuracy on held-out test set ===")
    print(f"Baseline network:      {accuracy_score(y_test, y_pred_base):.4f} "
          f"(F1={f1_score(y_test, y_pred_base):.4f})")
    print(f"Simplified surrogate:  {accuracy_score(y_test, y_pred_simp):.4f} "
          f"(F1={f1_score(y_test, y_pred_simp):.4f})")
    print(f"Fidelity (agreement):  {(y_pred_base == y_pred_simp).mean():.4f}")
    print(f"[basis={simplified.basis}]\n")

    t_base = baseline.forward_pass_latency(X_test).mean()
    t_simp = simplified.forward_pass_latency(X_test).mean()
    print("=== Latency (full test batch) ===")
    print(f"Baseline:   {t_base:.6f}s")
    print(f"Simplified: {t_simp:.6f}s")
    print(f"Speedup:    {t_base / t_simp:.2f}x "
          f"({'simplified faster' if t_simp < t_base else 'baseline faster'})\n")

    build_visualization(
        baseline, simplified, X_train, y_train, X_test, y_test,
        out_path=args.out, grid_res=args.grid_res, native_3d=True,
    )


if __name__ == "__main__":
    main()
