"""
model_adapters.py

Everything in boundary_simplifier.py only ever calls two methods on the
"baseline model": `.predict_proba(X) -> P(class=1)` and `.predict(X) ->
{0,1}`. That means the whole pipeline (distillation, autotuning, the fused
NumPy surrogate, the JS export) already works with ANY binary classifier
that exposes those two methods -- it was never actually specific to
MLPClassifier or to neural networks at all. This module makes that
explicit and removes the one piece of friction: not every classifier
exposes predict_proba in the same way.

ClassifierAdapter wraps:
  - sklearn classifiers with `.predict_proba` (RandomForest, GradientBoosting,
    KNeighbors, MLPClassifier, GaussianNB, SVC(probability=True), ...)
  - sklearn classifiers with only `.decision_function` (SVC without
    probability=True, SGDClassifier, Perceptron, LinearSVC, ...) -- mapped
    through a sigmoid to get a probability-like score
  - any plain callable `f(X) -> array of scores or probabilities`
  - anything else exposing `.predict` only, treated as a hard 0/1 oracle
    (predict_proba then returns exactly 0.0 or 1.0 -- the surrogate can
    still be fit, just against hard labels instead of soft ones)

This is what makes "does this work for any type of classifier" true in
practice, not just in principle.
"""

import time
import numpy as np
from scipy.special import expit


class ClassifierAdapter:
    def __init__(self, model):
        self.model = model
        if hasattr(model, "predict_proba"):
            self._mode = "predict_proba"
        elif hasattr(model, "decision_function"):
            self._mode = "decision_function"
        elif hasattr(model, "predict"):
            self._mode = "predict_only"
        elif callable(model):
            self._mode = "callable"
        else:
            raise TypeError(
                f"Don't know how to get predictions out of {type(model)!r} -- "
                "it needs .predict_proba, .decision_function, .predict, or to "
                "be directly callable."
            )

    def predict_proba(self, X):
        if self._mode == "predict_proba":
            out = self.model.predict_proba(X)
            return out[:, 1] if out.ndim == 2 else out
        elif self._mode == "decision_function":
            return expit(self.model.decision_function(X))
        elif self._mode == "predict_only":
            return self.model.predict(X).astype(float)
        else:  # callable
            out = np.asarray(self.model(X))
            return out[:, 1] if out.ndim == 2 else out

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def forward_pass_latency(self, X, n_repeats=20):
        self.predict_proba(X[:1])
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            self.predict_proba(X)
            times.append(time.perf_counter() - t0)
        return np.array(times)


def wrap(model):
    """Convenience: wrap anything, or pass through if it already looks
    like it satisfies the interface (has both predict_proba and predict
    already returning the expected shapes)."""
    if hasattr(model, "predict_proba") and hasattr(model, "predict") and hasattr(model, "forward_pass_latency"):
        return model  # already compatible (e.g. BaselineTextClassifier itself)
    return ClassifierAdapter(model)


if __name__ == "__main__":
    # Prove this works for classifiers that have nothing to do with neural
    # networks at all: RandomForest and a linear SVM (decision_function only).
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    from data_utils import load_text_classification_data
    from boundary_simplifier import autotune_surrogate

    X_train, X_test, y_train, y_test, vec, source = load_text_classification_data()
    Xtr = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    Xte = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    for name, clf in [
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=0)),
        ("LinearSVC (decision_function only)", LinearSVC(random_state=0, max_iter=5000)),
    ]:
        clf.fit(Xtr, y_train)
        baseline = wrap(clf)
        simplified, report = autotune_surrogate(baseline, Xtr, y_train,
                                                  max_accuracy_drop=0.01, verbose=False)
        base_acc = accuracy_score(y_test, baseline.predict(Xte))
        simp_acc = accuracy_score(y_test, simplified.predict(Xte))
        t_base = baseline.forward_pass_latency(Xte).mean()
        t_simp = simplified.forward_pass_latency(Xte).mean()
        print(f"{name:38s} baseline_acc={base_acc:.4f} simplified_acc={simp_acc:.4f} "
              f"speedup={t_base/t_simp:.2f}x  chosen={report['chosen_config']}")
