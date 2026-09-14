"""
Task 1: image classification on sklearn's `digits` dataset (8x8 grayscale,
10 classes, 1797 samples - a small, fast MNIST-like benchmark that's
reasonable to run repeatedly on a laptop).

Runs three systems on the identical train/test split and prints a
comparison table:
  A) NeuroSpace + exact backprop + structural plasticity  (main system)
  B) NeuroSpace + pure reward-modulated Hebbian + plasticity (brain-like, no backprop)
  C) Standard fixed MLP, same optimizer style, roughly matched size
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neurosim.network import NeuroSpace
from neurosim.learning import train_backprop, train_hebbian, evaluate_classification
from neurosim.baseline_mlp import MLP, train_mlp, evaluate_mlp


def load_data(n_train=300, n_test=200, seed=0):
    d = load_digits()
    X = d.data / 16.0  # normalize to [0,1]
    y = d.target
    Y = np.eye(10)[y]
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, train_size=n_train, test_size=n_test,
                                           stratify=y, random_state=seed)
    return Xtr, Ytr, Xte, Yte


def run(epochs=40, n_train=300, n_test=200, seed=0, lr=0.05, n_seed_hidden=24, seed_fanin=16):
    np.random.seed(seed)
    Xtr, Ytr, Xte, Yte = load_data(n_train, n_test, seed)
    n_in, n_out = Xtr.shape[1], Ytr.shape[1]
    results = {}

    print("=" * 70)
    print("A) NeuroSpace: backprop + structural plasticity")
    print("=" * 70)
    netA = NeuroSpace(n_in, n_out, output_z=10.0, hidden_activation="relu", seed=seed,
                       n_seed_hidden=n_seed_hidden, seed_fanin=seed_fanin)
    t0 = time.time()
    histA = train_backprop(netA, Xtr, Ytr, epochs=epochs, lr=lr, plasticity_every=2)
    trainA_time = time.time() - t0
    accA = evaluate_classification(netA, Xte, Yte)
    results["NeuroSpace (backprop+plasticity)"] = dict(
        acc=accA, train_time=trainA_time, params=netA.n_synapses(), hidden=netA.n_hidden())

    print("=" * 70)
    print("B) NeuroSpace: pure Hebbian (no backprop) + structural plasticity")
    print("=" * 70)
    netB = NeuroSpace(n_in, n_out, output_z=10.0, hidden_activation="relu", seed=seed,
                       n_seed_hidden=n_seed_hidden, seed_fanin=seed_fanin)
    t0 = time.time()
    histB = train_hebbian(netB, Xtr, Ytr, epochs=epochs, lr=lr, plasticity_every=2)
    trainB_time = time.time() - t0
    accB = evaluate_classification(netB, Xte, Yte)
    results["NeuroSpace (pure Hebbian)"] = dict(
        acc=accB, train_time=trainB_time, params=netB.n_synapses(), hidden=netB.n_hidden())

    print("=" * 70)
    print("C) Standard fixed MLP baseline")
    print("=" * 70)
    hidden_size = max(8, netA.n_hidden())  # roughly match capacity after growth
    mlp = MLP([n_in, hidden_size, n_out], seed=seed)
    t0 = time.time()
    histC = train_mlp(mlp, Xtr, Ytr, epochs=epochs, lr=lr)
    trainC_time = time.time() - t0
    accC = evaluate_mlp(mlp, Xte, Yte)
    results["Standard MLP"] = dict(acc=accC, train_time=trainC_time, params=mlp.n_params(), hidden=hidden_size)

    print("\n" + "=" * 70)
    print(f"RESULTS  (train={n_train} test={n_test} epochs={epochs})")
    print("=" * 70)
    print(f"{'System':40s} {'TestAcc':>8s} {'TrainTime(s)':>13s} {'#Params':>9s} {'#Hidden':>8s}")
    for name, r in results.items():
        print(f"{name:40s} {r['acc']*100:7.2f}% {r['train_time']:13.2f} {r['params']:9d} {r['hidden']:8d}")
    return results


if __name__ == "__main__":
    run()
