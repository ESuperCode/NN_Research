"""
Two ways to teach the network, so you can compare them directly:

1. train_backprop  - exact gradient descent through the dynamic DAG.
   This is what will actually work well on real tasks.

2. train_hebbian    - a reward-modulated Hebbian rule with NO
   backpropagation at all: weights move toward correlation with a global
   scalar reward signal (here: -error). This is much closer to how
   biological synaptic plasticity is theorized to work (three-factor
   Hebbian / node perturbation), and it is dramatically weaker at credit
   assignment in deep-ish sparse graphs. Included so you can see the gap
   yourself with real numbers, not just take my word for it.

Both call NeuroSpace.plasticity_step() periodically so structure keeps
evolving (growth/pruning/movement) no matter which weight rule is used.
"""
import numpy as np
import time


def train_backprop(net, X, Y, epochs=20, lr=0.05, plasticity_every=1, verbose=True):
    history = []
    for ep in range(epochs):
        t0 = time.time()
        perm = np.random.permutation(len(X))
        total_loss = 0.0
        for idx in perm:
            net.forward(X[idx])
            total_loss += net.backward(Y[idx], lr=lr)
        if plasticity_every and (ep + 1) % plasticity_every == 0:
            stats = net.plasticity_step()
        else:
            stats = {}
        history.append({"epoch": ep, "loss": total_loss / len(X), "time": time.time() - t0,
                         "n_hidden": net.n_hidden(), "n_syn": net.n_synapses(), **stats})
        if verbose:
            print(f"[backprop] epoch {ep+1}/{epochs} loss={history[-1]['loss']:.4f} "
                  f"hidden={net.n_hidden()} syn={net.n_synapses()} {stats}")
    return history


def train_hebbian(net, X, Y, epochs=20, lr=0.02, plasticity_every=1, verbose=True):
    """Reward-modulated Hebbian: after each sample, compute scalar reward
    r = -MSE, then nudge every synapse by lr * r * pre * post (three-factor
    rule). No error is propagated backward through the graph at all."""
    history = []
    for ep in range(epochs):
        t0 = time.time()
        perm = np.random.permutation(len(X))
        total_loss = 0.0
        for idx in perm:
            out = net.forward(X[idx])
            err = float(np.mean((out - Y[idx]) ** 2))
            total_loss += err
            reward = -err
            for n in net.neurons.values():
                if n.kind == "input":
                    continue
                for src_id, w in list(n.incoming.items()):
                    pre = net.neurons[src_id].last_output
                    post = n.last_output
                    n.incoming[src_id] = w + lr * reward * pre * post
        if plasticity_every and (ep + 1) % plasticity_every == 0:
            stats = net.plasticity_step()
        else:
            stats = {}
        history.append({"epoch": ep, "loss": total_loss / len(X), "time": time.time() - t0,
                         "n_hidden": net.n_hidden(), "n_syn": net.n_synapses(), **stats})
        if verbose:
            print(f"[hebbian ] epoch {ep+1}/{epochs} loss={history[-1]['loss']:.4f} "
                  f"hidden={net.n_hidden()} syn={net.n_synapses()} {stats}")
    return history


def evaluate_classification(net, X, Y_onehot):
    correct = 0
    for x, y in zip(X, Y_onehot):
        out = net.forward(x)
        if np.argmax(out) == np.argmax(y):
            correct += 1
    return correct / len(X)
