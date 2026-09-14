import numpy as np
import json
from sklearn.datasets import make_moons, make_circles, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from vector_nn import VectorNN, MLP
from train_utils import train_classifier, accuracy

RNG_SEED = 0


def make_vector_structured_task(n=2000, seed=0):
    """A task hand-designed to have genuine per-group vector structure, so we
    can see whether the norm-based squash nonlinearity has any real
    advantage when the data actually rewards reasoning about vector
    magnitude/direction as a unit, rather than each scalar independently.

    8 input dims = 4 groups of 2D vectors. Label depends on:
      - the norm of group 1 and group 2 vectors (radius-based, rotation
        invariant within each group) combined nonlinearly
      - plus pure noise dims (groups 3,4) that carry no signal
    A standard elementwise ReLU network *can* still learn this (it can
    learn to approximate sqrt(x^2+y^2) via enough ReLU units) but it is
    exactly the kind of computation squash does for free in one step.
    """
    rng = np.random.default_rng(seed)
    g1 = rng.normal(0, 1, size=(n, 2))
    g2 = rng.normal(0, 1, size=(n, 2))
    noise = rng.normal(0, 1, size=(n, 4))
    r1 = np.linalg.norm(g1, axis=1)
    r2 = np.linalg.norm(g2, axis=1)
    score = r1 ** 2 - r2 ** 2  # rotation-invariant nonlinear combination
    y = (score > np.median(score)).astype(int)
    X = np.concatenate([g1, g2, noise], axis=1)
    return X, y


def prep(X, y, test_size=0.25, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    return Xtr.astype(np.float64), Xte.astype(np.float64), ytr, yte


def widen_to_multiple(dim, group_size):
    """round dim up to nearest multiple of group_size (for fair architecture matching)"""
    import math
    return int(math.ceil(dim / group_size) * group_size)


def run_task(name, X, y, hidden_width=32, group_sizes=(1, 2, 4, 8), epochs=150, lr=0.1, seed=0):
    print(f"\n=== Task: {name} (n={len(y)}, dim={X.shape[1]}, classes={len(np.unique(y))}) ===")
    Xtr, Xte, ytr, yte = prep(X, y, seed=seed)
    n_classes = len(np.unique(y))
    input_dim = Xtr.shape[1]

    results = {}

    # Standard MLP baseline
    rng = np.random.default_rng(seed)
    mlp = MLP([input_dim, hidden_width, hidden_width], n_classes, rng=rng)
    res = train_classifier(mlp, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr,
                            rng=np.random.default_rng(seed + 100))
    results["MLP (scalar baseline)"] = res
    print(f"  MLP:            val_acc={res['final_val_acc']:.3f}  "
          f"params={res['n_params']}  time={res['elapsed']:.2f}s")

    for gs in group_sizes:
        if gs == 1:
            continue  # identical to MLP with elementwise -- skip, MLP baseline covers it
        rng = np.random.default_rng(seed)
        vnn = VectorNN([input_dim, hidden_width, hidden_width], group_size=gs,
                        n_classes=n_classes, nonlin="squash", rng=rng)
        res = train_classifier(vnn, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr,
                                rng=np.random.default_rng(seed + 100))
        label = f"VectorNN squash (group={gs})"
        results[label] = res
        print(f"  {label:28s} val_acc={res['final_val_acc']:.3f}  "
              f"params={res['n_params']}  time={res['elapsed']:.2f}s")

    return results


def main():
    all_results = {}

    # 1. make_moons - 2D, nonlinear boundary, no natural vector grouping beyond the 2 raw dims
    X, y = make_moons(n_samples=1500, noise=0.25, random_state=0)
    all_results["moons"] = run_task("make_moons", X, y, hidden_width=16, group_sizes=(2,), epochs=200, lr=0.3)

    # 2. make_circles
    X, y = make_circles(n_samples=1500, noise=0.15, factor=0.4, random_state=0)
    all_results["circles"] = run_task("make_circles", X, y, hidden_width=16, group_sizes=(2,), epochs=200, lr=0.3)

    # 3. sklearn digits - 64 dims, 10 classes, real (if small) image task
    digits = load_digits()
    all_results["digits"] = run_task("digits (8x8 images)", digits.data, digits.target,
                                      hidden_width=64, group_sizes=(2, 4, 8), epochs=150, lr=0.2)

    # 4. breast cancer - 30 tabular features, 2 classes
    bc = load_breast_cancer()
    all_results["breast_cancer"] = run_task("breast_cancer (tabular)", bc.data, bc.target,
                                             hidden_width=32, group_sizes=(2, 5), epochs=150, lr=0.1)

    # 5. Hand-built task with genuine vector/rotational structure
    X, y = make_vector_structured_task(n=2000, seed=0)
    all_results["vector_structured"] = run_task("synthetic vector-structured task", X, y,
                                                 hidden_width=16, group_sizes=(2, 4), epochs=200, lr=0.3)

    # Save a compact summary for plotting/report
    summary = {}
    for task, res in all_results.items():
        summary[task] = {model: {"val_acc": r["final_val_acc"], "n_params": r["n_params"],
                                  "elapsed": r["elapsed"]} for model, r in res.items()}
    with open("results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Also save full histories for plotting curves
    full = {}
    for task, res in all_results.items():
        full[task] = {model: r["history"] for model, r in res.items()}
    with open("results_history.json", "w") as f:
        json.dump(full, f, indent=2)

    print("\nSaved results_summary.json and results_history.json")


if __name__ == "__main__":
    main()
