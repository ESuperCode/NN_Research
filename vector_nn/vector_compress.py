"""
Vector squashing (capsule-style nonlinearity)
==============================================
squash(v) = (||v||^2 / (1 + ||v||^2)) * v / ||v||

Scales a vector's LENGTH into [0, 1) while leaving its DIRECTION unchanged.
Run this file directly to see a demo + a plot of the magnitude-gating curve.
"""

import numpy as np


def squash(v, eps=1e-8):
    """v: array of shape (..., d). Returns squashed vector, same shape."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    gate = n / (1.0 + n ** 2 + eps)   # scalar in [0, 1), grows then saturates
    return gate * v


def squash_backward(v, grad_out, eps=1e-8):
    """Backprop through squash.
    v:        input that was fed into squash, shape (..., d)
    grad_out: upstream gradient dL/d(squash(v)), same shape as v
    returns:  grad_in = dL/dv, same shape as v
    """
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n_safe = n + eps
    gate = n / (1.0 + n ** 2 + eps)
    gate_prime = (1.0 - n ** 2) / (1.0 + n ** 2 + eps) ** 2
    dot = np.sum(v * grad_out, axis=-1, keepdims=True)
    return gate * grad_out + (gate_prime / n_safe) * v * dot


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== 1. Basic behavior: direction preserved, length rescaled ===")
    v = np.array([3.0, 4.0])          # length 5
    out = squash(v)
    print(f"input  v = {v}, ||v|| = {np.linalg.norm(v):.3f}")
    print(f"output   = {out}, ||out|| = {np.linalg.norm(out):.3f}")
    print(f"same direction? cos angle = "
          f"{v @ out / (np.linalg.norm(v) * np.linalg.norm(out)):.6f}  (should be ~1.0)\n")

    print("=== 2. Magnitude gating curve: short vectors -> ~0, long vectors -> ~1 ===")
    for length in [0.1, 0.5, 1.0, 2.0, 5.0, 20.0]:
        v = np.array([length, 0.0])
        out = squash(v)
        print(f"  ||v|| = {length:6.2f}  ->  ||squash(v)|| = {np.linalg.norm(out):.4f}")
    print()

    print("=== 3. Gradient check (finite differences vs analytic) ===")
    v = rng.normal(size=5)
    grad_out = rng.normal(size=5)  # pretend this came from downstream

    analytic = squash_backward(v, grad_out)

    eps = 1e-6
    numeric = np.zeros_like(v)
    for i in range(len(v)):
        v_plus, v_minus = v.copy(), v.copy()
        v_plus[i] += eps
        v_minus[i] -= eps
        # directional derivative check: (loss(v+eps) - loss(v-eps)) / 2eps
        # where "loss" = dot(squash(v), grad_out)
        loss_plus = squash(v_plus) @ grad_out
        loss_minus = squash(v_minus) @ grad_out
        numeric[i] = (loss_plus - loss_minus) / (2 * eps)

    print(f"analytic grad = {np.round(analytic, 5)}")
    print(f"numeric  grad = {np.round(numeric, 5)}")
    print(f"max abs diff  = {np.max(np.abs(analytic - numeric)):.2e}  (should be tiny)\n")

    print("=== 4. Works on batches too: shape (batch, groups, dim) ===")
    batch = rng.normal(size=(4, 3, 2))  # 4 samples, 3 groups, 2D vectors each
    out = squash(batch)
    print(f"input shape  {batch.shape}")
    print(f"output shape {out.shape}")
    print(f"lengths in:  {np.round(np.linalg.norm(batch, axis=-1), 3)}")
    print(f"lengths out: {np.round(np.linalg.norm(out, axis=-1), 3)}")

    # Optional: plot the gating curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lengths = np.linspace(0, 10, 200)
        output_lengths = lengths ** 2 / (1 + lengths ** 2)  # ||squash(v)|| = g(n)*n
        plt.figure(figsize=(6, 4))
        plt.plot(lengths, output_lengths)
        plt.xlabel("input length ||v||")
        plt.ylabel("output length ||squash(v)||")
        plt.title("Squash magnitude-gating curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("squash_curve.png", dpi=150)
        print("\nSaved squash_curve.png")
    except ImportError:
        pass