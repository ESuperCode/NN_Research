"""
Task 3: image reconstruction / generation test. Trains each system as an
autoencoder (input = 8x8 digit image, target = the same image) through a
bottleneck, then reports reconstruction error and saves a before/after grid
so you can visually judge output quality - the real test of whether this is
"generating" something coherent versus just memorizing noise.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neurosim.network import NeuroSpace
from neurosim.learning import train_backprop
from neurosim.baseline_mlp import MLP, train_mlp


def run(epochs=60, n_train=300, n_test=40, seed=0, lr=0.08, out_path="/mnt/user-data/outputs/reconstruction.png"):
    d = load_digits()
    X = d.data / 16.0
    Xtr, Xte = train_test_split(X, train_size=n_train, test_size=n_test, random_state=seed)
    n = X.shape[1]

    print("Training NeuroSpace autoencoder (bottleneck via seeded hidden layer)...")
    net = NeuroSpace(n, n, output_z=6.0, hidden_activation="relu", output_activation="sigmoid",
                      seed=seed, n_seed_hidden=16, seed_fanin=n)  # 16-unit bottleneck < 64 inputs
    train_backprop(net, Xtr, Xtr, epochs=epochs, lr=lr, plasticity_every=5, verbose=False)
    recon_net = np.array([net.forward(x) for x in Xte])
    mse_net = float(np.mean((recon_net - Xte) ** 2))

    print("Training standard MLP autoencoder (same bottleneck size)...")
    mlp = MLP([n, 16, n], seed=seed)
    train_mlp(mlp, Xtr, Xtr, epochs=epochs, lr=lr, verbose=False)
    recon_mlp = np.array([mlp.forward(x) for x in Xte])
    mse_mlp = float(np.mean((recon_mlp - Xte) ** 2))

    print(f"\nReconstruction MSE (lower=better), bottleneck=16 units, test n={n_test}")
    print(f"  NeuroSpace (backprop+plasticity): {mse_net:.4f}   synapses={net.n_synapses()} hidden={net.n_hidden()}")
    print(f"  Standard MLP:                      {mse_mlp:.4f}   params={mlp.n_params()}")

    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i in range(8):
        axes[0, i].imshow(Xte[i].reshape(8, 8), cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(recon_net[i].reshape(8, 8), cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(recon_mlp[i].reshape(8, 8), cmap="gray"); axes[2, i].axis("off")
        axes[3, i].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=9); axes[1, 0].set_ylabel("NeuroSpace", fontsize=9)
    axes[2, 0].set_ylabel("MLP", fontsize=9)
    for ax, lbl in zip([axes[0, 0], axes[1, 0], axes[2, 0]], ["original", "NeuroSpace", "MLP"]):
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel(lbl, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Saved visual comparison to {out_path}")
    return dict(mse_net=mse_net, mse_mlp=mse_mlp)


if __name__ == "__main__":
    run()
