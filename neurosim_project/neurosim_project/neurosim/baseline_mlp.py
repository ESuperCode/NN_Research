"""A plain fixed-topology MLP, numpy only, same optimizer style (plain SGD)
as the NeuroSpace backprop path, so the comparison is about the *network
design* rather than about who has a fancier optimizer."""
import numpy as np
import time


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W = [rng.normal(0, np.sqrt(2 / sizes[i]), size=(sizes[i], sizes[i + 1]))
                   for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def _relu(self, x):
        return np.maximum(0, x)

    def _sig(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -60, 60)))

    def forward(self, x):
        acts = [x]
        a = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = self._sig(z) if i == len(self.W) - 1 else self._relu(z)
            acts.append(a)
        self._acts = acts
        return a

    def backward(self, target, lr=0.05, l2=1e-5):
        acts = self._acts
        out = acts[-1]
        loss = float(np.mean((out - target) ** 2))
        delta = 2 * (out - target) / len(out) * (out * (1 - out))
        for i in reversed(range(len(self.W))):
            a_prev = acts[i]
            gW = np.outer(a_prev, delta) + l2 * self.W[i]
            gb = delta
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
            self.W[i] -= lr * gW
            self.b[i] -= lr * gb
        return loss

    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)


def train_mlp(net, X, Y, epochs=20, lr=0.05, verbose=True):
    history = []
    for ep in range(epochs):
        t0 = time.time()
        perm = np.random.permutation(len(X))
        total_loss = 0.0
        for idx in perm:
            net.forward(X[idx])
            total_loss += net.backward(Y[idx], lr=lr)
        history.append({"epoch": ep, "loss": total_loss / len(X), "time": time.time() - t0})
        if verbose:
            print(f"[mlp     ] epoch {ep+1}/{epochs} loss={history[-1]['loss']:.4f}")
    return history


def evaluate_mlp(net, X, Y_onehot):
    correct = 0
    for x, y in zip(X, Y_onehot):
        out = net.forward(x)
        if np.argmax(out) == np.argmax(y):
            correct += 1
    return correct / len(X)
