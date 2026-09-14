"""Render the current NeuroSpace structure as a 3D scatter/line plot -
inputs, hidden neurons (colored/sized by activity), outputs, and synapses
as faint lines. Run after training to see what structure emerged."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def plot_network(net, out_path="/mnt/user-data/outputs/network_3d.png", max_synapses=1200):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    kind_style = {"input": ("#4C9AFF", 25), "hidden": ("#FF8B3D", 40), "output": ("#36B37E", 60)}
    for kind, (color, size) in kind_style.items():
        pts = np.array([net.neurons[nid].pos for nid in net.neurons if net.neurons[nid].kind == kind])
        if len(pts):
            if kind == "hidden":
                acts = np.array([net.neurons[nid].act_ema for nid in net.neurons if net.neurons[nid].kind == kind])
                sizes = 20 + 300 * (acts / (acts.max() + 1e-9))
            else:
                sizes = size
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=sizes, label=kind, depthshade=True)

    edges = []
    for n in net.neurons.values():
        for src_id, w in n.incoming.items():
            edges.append((net.neurons[src_id].pos, n.pos, abs(w)))
    edges.sort(key=lambda e: -e[2])
    edges = edges[:max_synapses]
    max_w = max((e[2] for e in edges), default=1.0)
    for p0, p1, w in edges:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color="gray", alpha=0.15 + 0.5 * (w / max_w), linewidth=0.6)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (input -> output depth)")
    ax.set_title(f"NeuroSpace structure: {net.n_hidden()} hidden neurons, {net.n_synapses()} synapses")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"Saved 3D network plot to {out_path}")


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from neurosim.network import NeuroSpace
    from neurosim.learning import train_backprop

    d = load_digits()
    X = d.data / 16.0
    y = d.target
    Y = np.eye(10)[y]
    Xtr, _, Ytr, _ = train_test_split(X, Y, train_size=300, random_state=0)
    net = NeuroSpace(64, 10, output_z=10.0, hidden_activation="relu", seed=0,
                      n_seed_hidden=24, seed_fanin=16)
    train_backprop(net, Xtr, Ytr, epochs=40, lr=0.05, plasticity_every=2, verbose=False)
    plot_network(net)
