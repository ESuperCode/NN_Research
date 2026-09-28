"""
synthetic_3d_datasets.py

Genuinely 3-dimensional, genuinely nonlinear two-class datasets, meant to
stress-test the boundary simplification method far harder than the mostly-
linearly-separable text topics do. Because these datasets are *natively*
3D, the visualization needs no PCA projection at all -- the extracted
surfaces are the EXACT 0.5-level boundary of each model, not an
approximate cross-section. This is the "make it actually hard" version.

Patterns included:
  - "helix":        two interleaved helices (like a 3D two-spirals problem;
                     non-convex, wraps around itself, classic hard case).
  - "swiss_roll":    two classes carved as bands along a rolled-up manifold
                     (the boundary is a spiral sheet embedded in 3D).
  - "xor":           3D checkerboard/XOR pattern (sign of x*y*z) -- boundary
                     is 8 disjoint octant regions, highly non-convex.
  - "spheres":       concentric shells -- boundary is two disjoint closed
                     surfaces (a topology polynomial surrogates should
                     struggle with more than the others).
"""

import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split


def make_double_helix(n_samples=1500, noise=0.3, turns=2.5, radius=4.0, random_state=42):
    rng = np.random.RandomState(random_state)
    n_each = n_samples // 2
    t = rng.uniform(0, 1, size=n_each) * turns * 2 * np.pi

    def helix(t, phase):
        x = radius * np.cos(t + phase)
        y = radius * np.sin(t + phase)
        z = (t / (turns * 2 * np.pi) - 0.5) * 2 * radius
        return np.stack([x, y, z], axis=1)

    X0 = helix(t, phase=0.0)
    X1 = helix(t, phase=np.pi)  # second helix, offset 180 degrees
    X = np.vstack([X0, X1])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n_each), np.ones(n_each)]).astype(int)
    return X, y


def make_swiss_roll_bands(n_samples=1500, noise=0.5, n_bands=4, random_state=42):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    # Alternate class along the roll parameter -> boundary is a spiral sheet.
    y = (np.floor(t / (t.max() / n_bands)) % 2).astype(int)
    return X, y


def make_xor_octants(n_samples=1500, noise=0.9, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-4, 4, size=(n_samples, 3))
    X += rng.normal(scale=noise * 0.15, size=X.shape)
    y = (np.sign(X[:, 0]) * np.sign(X[:, 1]) * np.sign(X[:, 2]) > 0).astype(int)
    return X, y


def make_concentric_spheres(n_samples=1500, noise=0.25, r_inner=2.0, r_outer=4.5, random_state=42):
    rng = np.random.RandomState(random_state)
    n_each = n_samples // 2

    def sphere_points(n, r):
        vec = rng.normal(size=(n, 3))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        return vec * r

    X0 = sphere_points(n_each, r_inner)
    X1 = sphere_points(n_each, r_outer)
    X = np.vstack([X0, X1])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n_each), np.ones(n_each)]).astype(int)
    return X, y


_GENERATORS = {
    "helix": make_double_helix,
    "swiss_roll": make_swiss_roll_bands,
    "xor": make_xor_octants,
    "spheres": make_concentric_spheres,
}


def load_3d_dataset(pattern="helix", n_samples=1500, test_size=0.25, random_state=42, **kwargs):
    if pattern not in _GENERATORS:
        raise ValueError(f"Unknown pattern {pattern!r}. Choose from {list(_GENERATORS)}.")
    X, y = _GENERATORS[pattern](n_samples=n_samples, random_state=random_state, **kwargs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, f"synthetic 3D pattern: {pattern}"


if __name__ == "__main__":
    for name in _GENERATORS:
        X_train, X_test, y_train, y_test, source = load_3d_dataset(name)
        print(f"{name:12s} train={X_train.shape} test={X_test.shape} "
              f"class balance train={np.bincount(y_train)}")
