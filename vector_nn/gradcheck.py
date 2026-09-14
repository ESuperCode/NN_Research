import numpy as np
from vector_nn import VectorLayer, VectorNN, softmax_cross_entropy

rng = np.random.default_rng(42)


def numeric_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f()
        x[idx] = orig - eps
        f_minus = f()
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def check_layer(nonlin):
    print(f"--- checking VectorLayer nonlin={nonlin} ---")
    n_in, d_in, n_out, d_out = 3, 4, 2, 5
    layer = VectorLayer(n_in, d_in, n_out, d_out, nonlin=nonlin, rng=rng)
    x = rng.normal(size=(2, n_in, d_in))
    target = rng.normal(size=(2, n_out, d_out))

    def loss_fn():
        out = layer.forward(x)
        return np.mean((out - target) ** 2)

    # analytic
    out = layer.forward(x)
    grad_out = 2 * (out - target) / out.size
    grad_x_analytic = layer.backward(grad_out)
    gW_analytic = layer.gW.copy()
    gb_analytic = layer.gb.copy()

    gW_numeric = numeric_grad(loss_fn, layer.W)
    gb_numeric = numeric_grad(loss_fn, layer.b)
    gx_numeric = numeric_grad(loss_fn, x)

    def relerr(a, b):
        return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-8)

    print("  dW relerr:", relerr(gW_analytic, gW_numeric))
    print("  db relerr:", relerr(gb_analytic, gb_numeric))
    print("  dx relerr:", relerr(grad_x_analytic, gx_numeric))


def check_full_network():
    print("--- checking full VectorNN + softmax CE ---")
    net = VectorNN([8, 6, 4], group_size=2, n_classes=3, nonlin="squash", rng=rng)
    X = rng.normal(size=(5, 8))
    y = rng.integers(0, 3, size=5)

    def loss_fn():
        logits = net.forward(X)
        loss, _ = softmax_cross_entropy(logits, y)
        return loss

    logits = net.forward(X)
    loss, grad_logits = softmax_cross_entropy(logits, y)
    net.backward(grad_logits)

    # check a couple of parameter tensors
    for li, layer in enumerate(net.layers):
        gW_numeric = numeric_grad(loss_fn, layer.W)
        relerr = np.max(np.abs(layer.gW - gW_numeric)) / (np.max(np.abs(gW_numeric)) + 1e-8)
        print(f"  layer {li} dW relerr:", relerr)

    gW_numeric = numeric_grad(loss_fn, net.readout_W)
    relerr = np.max(np.abs(net.g_readout_W - gW_numeric)) / (np.max(np.abs(gW_numeric)) + 1e-8)
    print("  readout dW relerr:", relerr)


if __name__ == "__main__":
    check_layer("elementwise")
    check_layer("squash")
    check_layer("none")
    check_full_network()
