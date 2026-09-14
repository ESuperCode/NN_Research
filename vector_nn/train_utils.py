import time
import numpy as np
from vector_nn import softmax_cross_entropy, mse_loss


def train_classifier(model, X_train, y_train, X_val, y_val, epochs=150, lr=0.1,
                      batch_size=32, rng=None, verbose=False):
    rng = rng or np.random.default_rng(0)
    n = X_train.shape[0]
    history = {"train_loss": [], "val_acc": [], "train_acc": []}
    t0 = time.time()
    for epoch in range(epochs):
        idx = rng.permutation(n)
        Xs, ys = X_train[idx], y_train[idx]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size]
            logits = model.forward(xb)
            loss, grad = softmax_cross_entropy(logits, yb)
            model.backward(grad)
            model.step(lr)
            epoch_loss += loss
            n_batches += 1
        history["train_loss"].append(epoch_loss / n_batches)
        if epoch % 5 == 0 or epoch == epochs - 1:
            train_acc = accuracy(model, X_train, y_train)
            val_acc = accuracy(model, X_val, y_val)
            history["train_acc"].append((epoch, train_acc))
            history["val_acc"].append((epoch, val_acc))
            if verbose:
                print(f"  epoch {epoch:4d}  loss {history['train_loss'][-1]:.4f}  "
                      f"train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")
    elapsed = time.time() - t0
    final_val_acc = accuracy(model, X_val, y_val)
    return {"history": history, "elapsed": elapsed, "final_val_acc": final_val_acc,
            "n_params": model.n_params()}


def accuracy(model, X, y):
    logits = model.forward(X)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def train_regressor(model, X_train, y_train, X_val, y_val, epochs=300, lr=0.05,
                     batch_size=32, rng=None, verbose=False):
    rng = rng or np.random.default_rng(0)
    n = X_train.shape[0]
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    for epoch in range(epochs):
        idx = rng.permutation(n)
        Xs, ys = X_train[idx], y_train[idx]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size]
            pred = model.forward(xb)
            loss, grad = mse_loss(pred, yb)
            model.backward(grad)
            model.step(lr)
            epoch_loss += loss
            n_batches += 1
        history["train_loss"].append(epoch_loss / n_batches)
        if epoch % 10 == 0 or epoch == epochs - 1:
            val_pred = model.forward(X_val)
            val_loss, _ = mse_loss(val_pred, y_val)
            history["val_loss"].append((epoch, val_loss))
    elapsed = time.time() - t0
    final_val_pred = model.forward(X_val)
    final_val_loss, _ = mse_loss(final_val_pred, y_val)
    return {"history": history, "elapsed": elapsed, "final_val_loss": final_val_loss,
            "n_params": model.n_params()}
