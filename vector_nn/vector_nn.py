"""
Vector Neural Network (VNN)
===========================

A neural net where each "neuron" carries a small vector (an array of k numbers)
instead of a single scalar, connected by learned matrices instead of learned
scalar weights, with a vector-aware nonlinearity.

Layer math
----------
Layer has n_in input groups of dim d_in, n_out output groups of dim d_out.
Flatten: x in R^(n_in*d_in), a single dense matrix W in R^(n_out*d_out, n_in*d_in)
does the linear map (this is the *general* form of "matrix instead of scalar
weight between every pair of neurons" -- it is mathematically identical to an
ordinary dense layer of width n_in*d_in -> n_out*d_out; there is no free lunch
from grouping the linear part alone).

The only place grouping can matter is the nonlinearity:
  - 'elementwise': relu/tanh applied per scalar. This makes the whole layer
    exactly equivalent to a standard MLP layer (used as ground-truth baseline
    and as a sanity check that the vector machinery is correct).
  - 'squash': capsule-style nonlinearity applied per GROUP:
        squash(v) = (||v||^2 / (1+||v||^2)) * v/||v||
    This mixes information across the components of a group when computing
    both the forward activation and the backward gradient, which a scalar
    nonlinearity cannot do. This is the actual candidate "new capability".

Both linear and nonlinear backward passes are derived analytically below and
verified against finite differences in gradcheck.py.
"""

import numpy as np


# ----------------------------------------------------------------------
# Nonlinearities
# ----------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def tanh_grad_from_output(y):
    return 1 - y ** 2


def squash(v, eps=1e-8):
    """v: (..., d). Capsule-style vector squashing nonlinearity.
    Returns squashed vector of same shape.
    """
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    g = n / (1.0 + n ** 2 + eps)  # scalar per group
    return g * v


def squash_backward(v, grad_out, eps=1e-8):
    """Given input v (..., d) and upstream gradient grad_out (..., d) w.r.t.
    squash(v), return grad_in (..., d) w.r.t. v.

    squash(v) = g(n) * v,  n = ||v||,  g(n) = n / (1 + n^2)
    Jacobian:  J = g(n) I + (g'(n)/n) * v v^T
    g'(n) = (1 - n^2) / (1 + n^2)^2
    grad_in = J^T grad_out = g(n) * grad_out + (g'(n)/n) * v * (v . grad_out)
    (J is symmetric, so J^T = J)
    """
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n_safe = n + eps
    g = n / (1.0 + n ** 2 + eps)
    gprime = (1.0 - n ** 2) / (1.0 + n ** 2 + eps) ** 2
    dot = np.sum(v * grad_out, axis=-1, keepdims=True)
    grad_in = g * grad_out + (gprime / n_safe) * v * dot
    return grad_in


# ----------------------------------------------------------------------
# Vector layer
# ----------------------------------------------------------------------

class VectorLayer:
    """One layer of the vector network.

    Input:  (batch, n_in, d_in)
    Output: (batch, n_out, d_out)
    """

    def __init__(self, n_in, d_in, n_out, d_out, nonlin="squash", rng=None):
        rng = rng or np.random.default_rng()
        self.n_in, self.d_in, self.n_out, self.d_out = n_in, d_in, n_out, d_out
        self.nonlin = nonlin
        fan_in = n_in * d_in
        # Xavier-ish init on the flattened linear map
        limit = np.sqrt(6.0 / (fan_in + n_out * d_out))
        self.W = rng.uniform(-limit, limit, size=(n_out * d_out, n_in * d_in))
        self.b = np.zeros(n_out * d_out)
        # grads
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        # cache
        self._x_flat = None
        self._z = None       # pre-activation, (batch, n_out, d_out)
        self._out = None

    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, x):
        # x: (batch, n_in, d_in)
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)              # (batch, n_in*d_in)
        z_flat = x_flat @ self.W.T + self.b         # (batch, n_out*d_out)
        z = z_flat.reshape(batch, self.n_out, self.d_out)

        if self.nonlin == "elementwise":
            out = relu(z)
        elif self.nonlin == "squash":
            out = squash(z)
        elif self.nonlin == "none":
            out = z
        else:
            raise ValueError(self.nonlin)

        self._x_flat = x_flat
        self._z = z
        self._out = out
        return out

    def backward(self, grad_out):
        # grad_out: (batch, n_out, d_out) -- dL/d(out)
        batch = grad_out.shape[0]

        if self.nonlin == "elementwise":
            grad_z = grad_out * relu_grad(self._z)
        elif self.nonlin == "squash":
            grad_z = squash_backward(self._z, grad_out)
        elif self.nonlin == "none":
            grad_z = grad_out
        else:
            raise ValueError(self.nonlin)

        grad_z_flat = grad_z.reshape(batch, -1)               # (batch, n_out*d_out)

        # NOTE: grad_out is assumed to already be dL/d(out) for a *mean-over-batch*
        # loss (i.e. the 1/batch factor is already baked in per-sample, as e.g.
        # softmax_cross_entropy's grad does). So we sum over the batch here,
        # NOT divide by batch again.
        self.gW = grad_z_flat.T @ self._x_flat
        self.gb = grad_z_flat.sum(axis=0)

        grad_x_flat = grad_z_flat @ self.W                    # (batch, n_in*d_in)
        grad_x = grad_x_flat.reshape(batch, self.n_in, self.d_in)
        return grad_x

    def step(self, lr):
        self.W -= lr * self.gW
        self.b -= lr * self.gb


class VectorNN:
    """Stack of VectorLayers. Final layer's output vectors are summed (or
    read out via a linear head) to produce class logits / regression output.
    """

    def __init__(self, layer_dims, group_size, n_classes, nonlin="squash", rng=None):
        """
        layer_dims: list of "widths" (total scalar count) for input, hidden..., last-hidden
                    e.g. [input_dim, h1, h2]
        group_size: vector dimension d used for all hidden groups (input layer
                    stays as group_size-d groups too, padded if needed)
        n_classes:  size of final readout
        """
        rng = rng or np.random.default_rng(0)
        self.group_size = group_size
        self.layers = []
        dims = layer_dims
        for i in range(len(dims) - 1):
            n_in = max(1, dims[i] // group_size)
            n_out = max(1, dims[i + 1] // group_size)
            self.layers.append(
                VectorLayer(n_in, group_size, n_out, group_size, nonlin=nonlin, rng=rng)
            )
        # readout layer: flatten last hidden groups -> n_classes, plain linear (no nonlin)
        last_width = self.layers[-1].n_out * self.layers[-1].d_out
        self.readout_W = rng.normal(0, np.sqrt(2.0 / last_width), size=(n_classes, last_width))
        self.readout_b = np.zeros(n_classes)
        self._last_hidden_flat = None
        self._input_shape = None

    def n_params(self):
        return sum(l.n_params() for l in self.layers) + self.readout_W.size + self.readout_b.size

    def _to_groups(self, X):
        # X: (batch, input_dim) -> pad to multiple of group_size -> (batch, n_in, group_size)
        batch, dim = X.shape
        n_groups = self.layers[0].n_in
        needed = n_groups * self.group_size
        if needed > dim:
            X = np.pad(X, ((0, 0), (0, needed - dim)))
        elif needed < dim:
            X = X[:, :needed]
        return X.reshape(batch, n_groups, self.group_size)

    def forward(self, X):
        h = self._to_groups(X)
        for layer in self.layers:
            h = layer.forward(h)
        flat = h.reshape(h.shape[0], -1)
        self._last_hidden_flat = flat
        logits = flat @ self.readout_W.T + self.readout_b
        return logits

    def backward(self, grad_logits):
        batch = grad_logits.shape[0]
        self.g_readout_W = grad_logits.T @ self._last_hidden_flat
        self.g_readout_b = grad_logits.sum(axis=0)
        grad_flat = grad_logits @ self.readout_W
        last = self.layers[-1]
        grad_h = grad_flat.reshape(batch, last.n_out, last.d_out)
        for layer in reversed(self.layers):
            grad_h = layer.backward(grad_h)
        return grad_h

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)
        self.readout_W -= lr * self.g_readout_W
        self.readout_b -= lr * self.g_readout_b


# ----------------------------------------------------------------------
# Standard scalar MLP baseline (independent implementation, same interface)
# ----------------------------------------------------------------------

class MLP:
    def __init__(self, layer_dims, n_classes, rng=None):
        rng = rng or np.random.default_rng(0)
        dims = layer_dims + [n_classes]
        self.Ws, self.bs = [], []
        for i in range(len(dims) - 1):
            limit = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
            self.Ws.append(rng.uniform(-limit, limit, size=(dims[i + 1], dims[i])))
            self.bs.append(np.zeros(dims[i + 1]))
        self._cache = []

    def n_params(self):
        return sum(W.size for W in self.Ws) + sum(b.size for b in self.bs)

    def forward(self, X):
        self._cache = []
        h = X
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = h @ W.T + b
            is_last = i == len(self.Ws) - 1
            out = z if is_last else relu(z)
            self._cache.append((h, z, out))
            h = out
        return h

    def backward(self, grad_out):
        self.gWs = [None] * len(self.Ws)
        self.gbs = [None] * len(self.bs)
        grad = grad_out
        for i in reversed(range(len(self.Ws))):
            h_in, z, out = self._cache[i]
            is_last = i == len(self.Ws) - 1
            grad_z = grad if is_last else grad * relu_grad(z)
            self.gWs[i] = grad_z.T @ h_in
            self.gbs[i] = grad_z.sum(axis=0)
            grad = grad_z @ self.Ws[i]
        return grad

    def step(self, lr):
        for i in range(len(self.Ws)):
            self.Ws[i] -= lr * self.gWs[i]
            self.bs[i] -= lr * self.gbs[i]


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------

def softmax_cross_entropy(logits, y):
    """y: int labels (batch,). Returns loss (scalar) and grad wrt logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / exp.sum(axis=1, keepdims=True)
    batch = logits.shape[0]
    loss = -np.log(probs[np.arange(batch), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(batch), y] -= 1
    grad /= batch
    return loss, grad


def mse_loss(pred, target):
    diff = pred - target
    loss = np.mean(diff ** 2)
    grad = 2 * diff / diff.size
    return loss, grad
