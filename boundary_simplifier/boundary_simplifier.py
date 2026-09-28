"""
boundary_simplifier.py  (v2 -- rebuilt for speed)

WHAT CHANGED AND WHY
---------------------
The v1 surrogate chained three separate scikit-learn objects at inference
time: PCA.transform -> PolynomialFeatures.transform -> LogisticRegression
.predict_proba. Each of those does its own input validation and array
bookkeeping. At the problem sizes here (small-to-medium networks, batches
of a few hundred points) that FIXED per-call Python overhead dominates the
actual floating-point cost -- which is exactly why the v1 surrogate was
often *slower* than the baseline network despite doing less arithmetic.

v2 fixes this by algebraically folding the whole pipeline into a single
closed-form function evaluated with two or three raw NumPy operations and
zero scikit-learn calls at inference time:

  - basis="poly", degree=2 (the default, and the fastest): PCA is linear
    and the degree-2 polynomial-logistic decision function is an exact
    quadratic form in the PCA-reduced coordinates z. Composing them gives

        score(x) = c + a . z + z^T W z ,   z = P^T (x - mean)

    computed as ONE projection matmul (X @ P) plus ONE (k x k) quadratic
    form per row -- no PolynomialFeatures object, no LogisticRegression
    object, at inference time. This is mathematically exact (verified
    against sklearn's own PolynomialFeatures+LogisticRegression output to
    float precision), not an approximation of the same formula.
  - basis="poly", degree>2: same idea, but the exponent combinations are
    precomputed once at conversion time and evaluated via vectorized
    NumPy indexing/products at inference time instead of calling
    PolynomialFeatures.transform().
  - basis="rbf": the random Fourier feature map and the logistic weights
    are extracted into raw arrays once; inference is one projection
    matmul, one (matmul + cosine), and one dot product.

Deliberately NOT done: folding the PCA projection itself away (i.e.
expressing everything as a single d x d quadratic form in the ORIGINAL
input space). That would look elegant but is wrong for exactly the
dimensionality this project cares about: a d x d matrix costs O(d^2) per
sample, which for TF-IDF text (d in the thousands) is far more expensive
than staying in the k-dimensional PCA space (k in the tens) and paying
only the O(d*k) projection cost. Keeping the low-rank structure explicit
is what makes this fast at realistic text dimensionality, not despite it.

ACCURACY-CONSTRAINED AUTOTUNING
--------------------------------
`autotune_surrogate` tries a small set of increasingly expensive configs
(starting from the cheapest: few PCA components, degree-2 poly) against an
internal validation split carved out of the TRAINING data (never the held-
out test set -- that would invalidate any test-set numbers reported
afterward). It picks the fastest config whose validation accuracy is
within `max_accuracy_drop` of the baseline's own validation accuracy, and
falls back to the least-bad option (with a clear warning) if none qualify.
"""

import time
import numpy as np
from scipy import sparse
from scipy.special import expit  # numerically stable sigmoid
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split


def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=np.float64)


# ---------------------------------------------------------------------------
# The fast, fused-NumPy surrogate
# ---------------------------------------------------------------------------

class SimplifiedBoundaryModel:
    """
    Closed-form surrogate, evaluated with pure NumPy (no scikit-learn calls
    in predict_proba). Internally holds one of three fused representations,
    chosen by `basis`:

      "poly2"  -- exact quadratic form: score = c + Z@a + sum((Z@W)*Z, axis=1)
      "polyN"  -- degree>2 polynomial via precomputed exponent indices
      "rbf"    -- random-Fourier-feature linear model
    """

    def __init__(self, mean, P, basis, params, n_components):
        self.mean = mean          # (d,) PCA mean, or None if skip_pca
        self.P = P                # (d,k) PCA components^T, or None if skip_pca
        self.basis = basis
        self.params = params      # dict of the fused arrays for this basis
        self.n_components = n_components

    # -- projection shared by every basis --
    def _project(self, X):
        Xd = _to_dense(X)
        if self.P is None:
            return Xd
        return (Xd - self.mean) @ self.P

    def predict_proba(self, X):
        Z = self._project(X)
        if self.basis == "poly2":
            a, W, c = self.params["a"], self.params["W"], self.params["c"]
            quad = np.einsum("ij,jk,ik->i", Z, W, Z)
            score = c + Z @ a + quad
        elif self.basis == "polyN":
            idx, coef, c = self.params["idx"], self.params["coef"], self.params["c"]
            # idx: (n_features, degree) array of column indices (padded with
            # -1, treated as "multiply by 1") describing each monomial.
            feats = np.ones((Z.shape[0], idx.shape[0]))
            for col in range(idx.shape[1]):
                mask = idx[:, col] >= 0
                if mask.any():
                    feats[:, mask] *= Z[:, idx[mask, col]]
            score = c + feats @ coef
        elif self.basis == "rbf":
            Omega, b, w, c, scale = (self.params["Omega"], self.params["b"],
                                      self.params["w"], self.params["c"], self.params["scale"])
            phi = np.cos(Z @ Omega + b) * scale
            score = c + phi @ w
        else:
            raise ValueError(self.basis)
        return expit(score)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def as_formula_string(self):
        if self.basis == "poly2":
            k = self.n_components
            return (f"score(z) = {self.params['c']:.4f} + a.z + z^T W z   "
                     f"(z = PCA-projected input, a in R^{k}, W a {k}x{k} symmetric matrix)\n"
                     f"P(class=1) = sigmoid(score(z))")
        elif self.basis == "polyN":
            return (f"score(z) = {self.params['c']:.4f} + sum_m coef_m * monomial_m(z), "
                     f"{len(self.params['coef'])} monomials up to the fitted degree\n"
                     f"P(class=1) = sigmoid(score(z))")
        else:
            k = len(self.params["b"])
            return (f"score(z) = {self.params['c']:.4f} + sum_{{i=1}}^{{{k}}} w_i * cos(omega_i.z + b_i)\n"
                     f"P(class=1) = sigmoid(score(z))")

    def n_flops_per_sample(self):
        """Rough FLOP estimate per sample, for reporting model 'compute size'
        independent of wall-clock (which also reflects call overhead)."""
        d = self.P.shape[0] if self.P is not None else self.n_components
        k = self.n_components
        proj = 2 * d * k if self.P is not None else 0
        if self.basis == "poly2":
            return proj + 2 * k * k + 2 * k
        elif self.basis == "polyN":
            return proj + self.params["idx"].size + len(self.params["coef"])
        else:
            D = len(self.params["b"])
            return proj + 2 * k * D + D

    def forward_pass_latency(self, X, n_repeats=20):
        self.predict_proba(X[:1] if hasattr(X, "shape") and X.shape[0] > 0 else X)
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            self.predict_proba(X)
            times.append(time.perf_counter() - t0)
        return np.array(times)


# ---------------------------------------------------------------------------
# Conversion: baseline network -> fused closed-form SimplifiedBoundaryModel
# ---------------------------------------------------------------------------

def _sample_synthetic_points(X_train_dense, n_synthetic_per_point, jitter_std,
                              clip_non_negative, rng):
    """
    Generates synthetic points to query the teacher network on, IN THE
    ORIGINAL FEATURE SPACE (not through a low-rank PCA reconstruction --
    see the note below for why that matters).

    Uses mixup (convex combinations of random pairs of real training
    points) plus small multiplicative jitter restricted to each sample's
    already-nonzero entries. Both choices matter for sparse, high-
    dimensional data such as TF-IDF text (a few dozen nonzero features out
    of several thousand):

      - A convex combination of two real (non-negative) points is itself
        non-negative and stays inside the training data's convex hull by
        construction -- it can't wander into a region the network has
        never seen anything resembling.
      - Restricting jitter to already-nonzero entries (rather than
        perturbing all D dimensions with dense Gaussian noise) avoids
        injecting noise into thousands of structurally-zero dimensions,
        which for very sparse data can swamp the handful of dimensions
        that actually carry signal and collapse the teacher's predictions
        to a single class regardless of the true input (this was a real,
        reproduced failure mode of the earlier PCA-jitter-and-reconstruct
        approach: on sparse ~3000-dim data, reconstructing from a 6-
        component PCA plus jitter destroyed the signal in every synthetic
        sample, causing every candidate surrogate to see single-class
        teacher labels and fail to fit).
    """
    n = X_train_dense.shape[0]
    n_synth = n * n_synthetic_per_point
    i_idx = rng.randint(0, n, n_synth)
    j_idx = rng.randint(0, n, n_synth)
    alpha = rng.uniform(0.15, 0.85, size=(n_synth, 1))
    X_mix = alpha * X_train_dense[i_idx] + (1 - alpha) * X_train_dense[j_idx]

    nonzero_mask = X_mix != 0
    noise = rng.normal(scale=jitter_std * 0.3, size=X_mix.shape)
    X_mix = X_mix * (1 + noise * nonzero_mask)

    if clip_non_negative:
        X_mix = np.clip(X_mix, 0, None)
    return np.vstack([X_train_dense, X_mix])


def _fit_pca_and_sample(baseline_model, X_train, n_components, skip_pca,
                          n_synthetic_per_point, jitter_std, clip_non_negative,
                          random_state):
    rng = np.random.RandomState(random_state)
    X_train_dense = _to_dense(X_train)

    if skip_pca:
        mean, P = None, None
        n_components = X_train_dense.shape[1]
    else:
        n_components = min(n_components, X_train_dense.shape[1], X_train_dense.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(X_train_dense)
        mean, P = pca.mean_, pca.components_.T  # P: (d, k)

    X_all = _sample_synthetic_points(X_train_dense, n_synthetic_per_point, jitter_std,
                                      clip_non_negative, rng)
    teacher_hard = (baseline_model.predict_proba(X_all) >= 0.5).astype(int)

    if len(np.unique(teacher_hard)) < 2:
        raise RuntimeError(
            f"Teacher network predicted a single class across all {X_all.shape[0]} "
            f"sampled points (n_components={n_components}). This usually means "
            "jitter_std is too large relative to this data's scale for this "
            "particular config -- autotune_surrogate will skip this candidate "
            "and try the next one."
        )

    Z_all = (X_all - mean) @ P if P is not None else X_all
    return mean, P, Z_all, teacher_hard, n_components


def _build_poly2_fused(Z_all, teacher_hard):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Zp = poly.fit_transform(Z_all)
    logreg = LogisticRegression(max_iter=5000).fit(Zp, teacher_hard)

    k = Z_all.shape[1]
    names = poly.get_feature_names_out([f"z{i}" for i in range(k)])
    coefs = logreg.coef_[0]
    a = np.zeros(k)
    W = np.zeros((k, k))
    for name, coef in zip(names, coefs):
        if "^2" in name:
            i = int(name.split("^")[0][1:])
            W[i, i] = coef
        elif " " in name:
            i, j = (int(p[1:]) for p in name.split(" "))
            W[i, j] = W[j, i] = coef / 2.0
        else:
            a[int(name[1:])] = coef
    return {"a": a, "W": W, "c": float(logreg.intercept_[0])}


def _build_polyN_fused(Z_all, teacher_hard, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Zp = poly.fit_transform(Z_all)
    logreg = LogisticRegression(max_iter=5000).fit(Zp, teacher_hard)

    powers = poly.powers_  # (n_features, k), exponent of each input dim per monomial
    max_deg = int(powers.sum(axis=1).max())
    idx = np.full((powers.shape[0], max_deg), -1, dtype=int)
    for f in range(powers.shape[0]):
        cols = []
        for dim, p in enumerate(powers[f]):
            cols += [dim] * int(p)
        idx[f, :len(cols)] = cols
    return {"idx": idx, "coef": logreg.coef_[0], "c": float(logreg.intercept_[0])}


def _build_rbf_fused(Z_all, teacher_hard, n_rbf_features, rbf_gamma, random_state):
    rbf = RBFSampler(gamma=rbf_gamma, n_components=n_rbf_features, random_state=random_state)
    Zf = rbf.fit_transform(Z_all)
    logreg = LogisticRegression(max_iter=5000).fit(Zf, teacher_hard)
    scale = np.sqrt(2.0 / n_rbf_features)
    return {"Omega": rbf.random_weights_, "b": rbf.random_offset_,
            "w": logreg.coef_[0], "c": float(logreg.intercept_[0]), "scale": scale}


def convert_to_simplified_boundary(
    baseline_model,
    X_train,
    n_components=20,
    n_synthetic_per_point=5,
    jitter_std=0.5,
    basis="poly",
    poly_degree=2,
    n_rbf_features=200,
    rbf_gamma=0.5,
    clip_non_negative=True,
    skip_pca=False,
    random_state=42,
):
    """Same signature/semantics as v1 -- drop-in replacement. `basis="poly"`
    with `poly_degree=2` (the default) uses the exact closed-form quadratic
    fold; `poly_degree>2` uses the precomputed-exponent fused path;
    `basis="rbf"` uses the fused random-Fourier-feature path."""
    mean, P, Z_all, teacher_hard, n_components = _fit_pca_and_sample(
        baseline_model, X_train, n_components, skip_pca,
        n_synthetic_per_point, jitter_std, clip_non_negative, random_state
    )

    if basis == "poly" and poly_degree == 2:
        params = _build_poly2_fused(Z_all, teacher_hard)
        fused_basis = "poly2"
    elif basis == "poly":
        params = _build_polyN_fused(Z_all, teacher_hard, poly_degree)
        fused_basis = "polyN"
    elif basis == "rbf":
        params = _build_rbf_fused(Z_all, teacher_hard, n_rbf_features, rbf_gamma, random_state)
        fused_basis = "rbf"
    else:
        raise ValueError(f"Unknown basis {basis!r}; choose 'poly' or 'rbf'.")

    return SimplifiedBoundaryModel(mean, P, fused_basis, params, n_components)


# ---------------------------------------------------------------------------
# Accuracy-constrained autotuner: picks the FASTEST config that stays within
# `max_accuracy_drop` of the baseline's own accuracy, measured on an internal
# validation split carved out of TRAINING data (never the held-out test set).
# ---------------------------------------------------------------------------

_CANDIDATE_CONFIGS = [
    # Ordered cheapest-first; the search stops at the first (fastest) config
    # that meets the accuracy bar, so cheaper candidates are tried first.
    dict(basis="poly", n_components=6,  poly_degree=2),
    dict(basis="poly", n_components=10, poly_degree=2),
    dict(basis="poly", n_components=15, poly_degree=2),
    dict(basis="poly", n_components=20, poly_degree=2),
    dict(basis="poly", n_components=10, poly_degree=3),
    dict(basis="poly", n_components=15, poly_degree=3),
    dict(basis="poly", n_components=20, poly_degree=4),
    dict(basis="rbf",  n_components=10, n_rbf_features=100, rbf_gamma=0.3),
    dict(basis="rbf",  n_components=15, n_rbf_features=200, rbf_gamma=0.3),
    dict(basis="rbf",  n_components=20, n_rbf_features=300, rbf_gamma=0.3),
]


def autotune_surrogate(baseline_model, X_train, y_train, max_accuracy_drop=0.001,
                        skip_pca=False, clip_non_negative=True, random_state=42,
                        verbose=True):
    """
    Returns (best_model, report) where report is a list of dicts describing
    every candidate tried, in order, with its validation accuracy delta and
    measured latency -- so the choice is auditable, not a black box.

    max_accuracy_drop is in ACCURACY FRACTION (0.001 = 0.1 percentage
    points), matching "no more than 0.1% worse than baseline".
    """
    X_train_dense = _to_dense(X_train)
    y_train_arr = np.asarray(y_train)
    Xt, Xv, yt, yv = train_test_split(X_train_dense, y_train_arr, test_size=0.25,
                                        random_state=random_state, stratify=y_train_arr)

    baseline_val_acc = float((baseline_model.predict(Xv) == yv).mean())
    report = []
    qualifying = []

    for cfg in _CANDIDATE_CONFIGS:
        try:
            model = convert_to_simplified_boundary(
                baseline_model, Xt, skip_pca=skip_pca, clip_non_negative=clip_non_negative,
                random_state=random_state, **cfg
            )
        except Exception as e:
            report.append({**cfg, "status": f"failed: {e!r}"})
            continue

        val_acc = float((model.predict(Xv) == yv).mean())
        lat = model.forward_pass_latency(Xv, n_repeats=15).mean()
        acc_drop = baseline_val_acc - val_acc
        meets_bar = acc_drop <= max_accuracy_drop

        row = {**cfg, "val_accuracy": val_acc, "baseline_val_accuracy": baseline_val_acc,
               "accuracy_drop": acc_drop, "meets_accuracy_bar": meets_bar,
               "val_latency_s": lat}
        report.append(row)
        if meets_bar:
            qualifying.append((lat, cfg, model, row))

    if qualifying:
        qualifying.sort(key=lambda t: t[0])
        best_lat, best_cfg, _, best_row = qualifying[0]
        status = "OK"
    else:
        # Nothing met the accuracy bar -- fall back to whichever candidate
        # had the smallest accuracy drop, and say so plainly.
        candidates_with_acc = [r for r in report if "accuracy_drop" in r]
        if not candidates_with_acc:
            failures = "\n".join(f"  {cfg}: {cfg.get('status', 'unknown error')}"
                                  for cfg in report)
            raise RuntimeError(
                "autotune_surrogate: every single candidate configuration failed to "
                "fit (none produced a usable accuracy number). This points to a "
                "data/preprocessing issue rather than a hyperparameter problem -- "
                "the most common cause is jitter_std being poorly scaled for this "
                "data (see _sample_synthetic_points). Per-candidate errors:\n"
                + failures
            )
        best_row = min(candidates_with_acc, key=lambda r: r["accuracy_drop"])
        best_cfg = {k: best_row[k] for k in ("basis", "n_components")
                    if k in best_row}
        for k in ("poly_degree", "n_rbf_features", "rbf_gamma"):
            if k in best_row:
                best_cfg[k] = best_row[k]
        status = "NO CONFIG MET THE ACCURACY BAR -- using the least-bad option"

    # Refit the chosen config on the FULL training set (the search above
    # only used a 75% sub-split) before returning it for real evaluation.
    final_model = convert_to_simplified_boundary(
        baseline_model, X_train_dense, skip_pca=skip_pca, clip_non_negative=clip_non_negative,
        random_state=random_state, **best_cfg
    )

    if verbose:
        print(f"  [autotune] baseline val accuracy: {baseline_val_acc:.4f}  "
              f"(bar: surrogate must be >= {baseline_val_acc - max_accuracy_drop:.4f})")
        for r in report:
            if "val_accuracy" in r:
                tag = "OK " if r["meets_accuracy_bar"] else "   "
                print(f"  [autotune] {tag} {r['basis']:5s} nc={r['n_components']:<3d} "
                      f"val_acc={r['val_accuracy']:.4f} (drop={r['accuracy_drop']:+.4f}) "
                      f"val_latency={r['val_latency_s']*1000:.3f}ms")
        print(f"  [autotune] chosen: {best_cfg}  [{status}]")

    return final_model, {"status": status, "baseline_val_accuracy": baseline_val_acc,
                          "chosen_config": best_cfg, "candidates": report}


if __name__ == "__main__":
    from data_utils import load_text_classification_data
    from baseline_model import train_baseline
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test, vec, source = load_text_classification_data()
    baseline = train_baseline(X_train, y_train)

    simplified, tuning_report = autotune_surrogate(baseline, X_train, y_train,
                                                     max_accuracy_drop=0.001)

    base_acc = accuracy_score(y_test, baseline.predict(X_test))
    simp_acc = accuracy_score(y_test, simplified.predict(_to_dense(X_test)))
    print(f"\nSource: {source}")
    print(f"Baseline accuracy:   {base_acc:.4f}")
    print(f"Simplified accuracy: {simp_acc:.4f}")
    print()
    print(simplified.as_formula_string())
