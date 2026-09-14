"""
Task 2: text topic classification. Uses a small, self-contained synthetic
corpus (built from topic-associated vocabulary + templates) instead of a
downloaded dataset, since this environment has no general internet access.
Bag-of-words -> TF-ish vector -> classify into one of 4 topics.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from neurosim.network import NeuroSpace
from neurosim.learning import train_backprop, train_hebbian, evaluate_classification
from neurosim.baseline_mlp import MLP, train_mlp, evaluate_mlp

TOPICS = {
    "sports": ["team", "score", "game", "player", "coach", "win", "match", "ball",
               "stadium", "league", "goal", "champion", "tournament", "referee"],
    "weather": ["rain", "sunny", "storm", "temperature", "forecast", "cloud", "wind",
                "snow", "humid", "climate", "thunder", "degrees", "weather", "sky"],
    "food": ["recipe", "dish", "flavor", "restaurant", "chef", "meal", "ingredient",
             "kitchen", "spicy", "delicious", "cook", "menu", "taste", "cuisine"],
    "technology": ["software", "computer", "algorithm", "device", "startup", "data",
                   "internet", "app", "server", "code", "chip", "network", "robot", "ai"],
}
FILLERS = ["the", "a", "was", "very", "really", "today", "yesterday", "this", "that",
           "amazing", "terrible", "interesting", "new", "old", "big", "small", "quickly"]


def make_corpus(rng, n_per_topic=80, sent_len=8):
    topics = list(TOPICS.keys())
    texts, labels = [], []
    for ti, topic in enumerate(topics):
        vocab = TOPICS[topic]
        for _ in range(n_per_topic):
            n_topic_words = rng.integers(2, 5)
            words = list(rng.choice(vocab, size=n_topic_words, replace=True))
            n_fillers = sent_len - n_topic_words
            words += list(rng.choice(FILLERS, size=max(0, n_fillers), replace=True))
            rng.shuffle(words)
            texts.append(" ".join(words))
            labels.append(ti)
    return texts, np.array(labels), topics


def run(epochs=25, seed=0, lr=0.05):
    rng = np.random.default_rng(seed)
    texts, labels, topics = make_corpus(rng)
    vec = CountVectorizer(max_features=60, binary=True)
    X = vec.fit_transform(texts).toarray().astype(np.float64)
    Y = np.eye(len(topics))[labels]

    idx = rng.permutation(len(X))
    split = int(0.7 * len(X))
    tr, te = idx[:split], idx[split:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    n_in, n_out = X.shape[1], len(topics)

    results = {}
    print(f"Corpus: {len(X)} samples, {n_in} vocab features, {n_out} topics")

    netA = NeuroSpace(n_in, n_out, output_z=8.0, hidden_activation="relu", seed=seed,
                       n_seed_hidden=20, seed_fanin=12)
    t0 = time.time()
    train_backprop(netA, Xtr, Ytr, epochs=epochs, lr=lr, plasticity_every=3, verbose=False)
    results["NeuroSpace (backprop+plasticity)"] = dict(
        acc=evaluate_classification(netA, Xte, Yte), time=time.time() - t0,
        params=netA.n_synapses(), hidden=netA.n_hidden())

    netB = NeuroSpace(n_in, n_out, output_z=8.0, hidden_activation="relu", seed=seed,
                       n_seed_hidden=20, seed_fanin=12)
    t0 = time.time()
    train_hebbian(netB, Xtr, Ytr, epochs=epochs, lr=lr, plasticity_every=3, verbose=False)
    results["NeuroSpace (pure Hebbian)"] = dict(
        acc=evaluate_classification(netB, Xte, Yte), time=time.time() - t0,
        params=netB.n_synapses(), hidden=netB.n_hidden())

    mlp = MLP([n_in, 20, n_out], seed=seed)
    t0 = time.time()
    train_mlp(mlp, Xtr, Ytr, epochs=epochs, lr=lr, verbose=False)
    results["Standard MLP"] = dict(acc=evaluate_mlp(mlp, Xte, Yte), time=time.time() - t0,
                                    params=mlp.n_params(), hidden=20)

    print(f"\n{'System':40s} {'TestAcc':>8s} {'Time(s)':>9s} {'#Params':>9s} {'#Hidden':>8s}")
    for name, r in results.items():
        print(f"{name:40s} {r['acc']*100:7.2f}% {r['time']:9.2f} {r['params']:9d} {r['hidden']:8d}")
    return results


if __name__ == "__main__":
    run()
