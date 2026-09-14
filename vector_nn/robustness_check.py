import numpy as np
from sklearn.datasets import make_circles
from vector_nn import VectorNN, MLP
from train_utils import train_classifier
from experiments import make_vector_structured_task, prep

def repeat(task_name, X, y, hidden_width, group_size, epochs, lr, n_seeds=5):
    mlp_accs, vnn_accs = [], []
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte = prep(X, y, seed=seed)
        n_classes = len(np.unique(y))
        input_dim = Xtr.shape[1]

        mlp = MLP([input_dim, hidden_width, hidden_width], n_classes, rng=np.random.default_rng(seed))
        r1 = train_classifier(mlp, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr,
                               rng=np.random.default_rng(seed + 100))
        mlp_accs.append(r1["final_val_acc"])

        vnn = VectorNN([input_dim, hidden_width, hidden_width], group_size=group_size,
                        n_classes=n_classes, nonlin="squash", rng=np.random.default_rng(seed))
        r2 = train_classifier(vnn, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr,
                               rng=np.random.default_rng(seed + 100))
        vnn_accs.append(r2["final_val_acc"])

    print(f"{task_name}: MLP {np.mean(mlp_accs):.3f} +/- {np.std(mlp_accs):.3f}  "
          f"(runs: {[round(a,3) for a in mlp_accs]})")
    print(f"{task_name}: VNN {np.mean(vnn_accs):.3f} +/- {np.std(vnn_accs):.3f}  "
          f"(runs: {[round(a,3) for a in vnn_accs]})")


if __name__ == "__main__":
    X, y = make_circles(n_samples=1500, noise=0.15, factor=0.4, random_state=0)
    repeat("circles (group=2)", X, y, hidden_width=16, group_size=2, epochs=200, lr=0.3)

    print()
    X, y = make_vector_structured_task(n=2000, seed=0)
    repeat("vector_structured (group=4)", X, y, hidden_width=16, group_size=4, epochs=200, lr=0.3)
