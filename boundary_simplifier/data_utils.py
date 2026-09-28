"""
data_utils.py

Loads a real binary text-classification dataset (20 Newsgroups, two categories)
when network access is available. Falls back to a bundled synthetic two-topic
corpus so the rest of the pipeline can be developed/tested offline. The fallback
is clearly logged -- it is not a substitute for the real benchmark, just a way
to exercise the code without internet access.
"""

import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


CATEGORIES = ["sci.space", "rec.sport.hockey"]


def _load_real_20newsgroups():
    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(
        subset="train", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )
    test = fetch_20newsgroups(
        subset="test", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )
    X_text = list(train.data) + list(test.data)
    y = list(train.target) + list(test.target)
    return X_text, np.array(y), "20 Newsgroups (sci.space vs rec.sport.hockey)"


# --- Offline fallback ---------------------------------------------------
# Template-based synthetic corpus across two topics with overlapping,
# noisy vocabulary. This is NOT a real dataset -- it exists only so the
# pipeline can be exercised without network access. Swap in
# _load_real_20newsgroups (or any other text corpus) for real experiments.

_SPACE_VOCAB = [
    "orbit", "satellite", "rocket", "launch", "telescope", "planet", "mission",
    "shuttle", "astronaut", "gravity", "nebula", "spacecraft", "booster",
    "trajectory", "lunar", "mars", "payload", "propulsion", "vacuum", "station",
]
_HOCKEY_VOCAB = [
    "puck", "goalie", "rink", "slapshot", "playoff", "penalty", "forward",
    "defenseman", "referee", "overtime", "roster", "powerplay", "faceoff",
    "coach", "season", "trade", "contract", "arena", "captain", "line",
]
_FILLER = [
    "the", "team", "last", "week", "reported", "said", "according", "new",
    "plan", "program", "budget", "officials", "experts", "believe", "will",
    "could", "should", "was", "were", "yesterday", "today", "story", "update",
]


def _make_synthetic_doc(vocab, rng, n_words=40):
    words = []
    for _ in range(n_words):
        if rng.random() < 0.55:
            words.append(rng.choice(vocab))
        else:
            words.append(rng.choice(_FILLER))
    return " ".join(words)


def _load_synthetic(n_per_class=400, seed=0):
    rng = random.Random(seed)
    docs, labels = [], []
    for _ in range(n_per_class):
        docs.append(_make_synthetic_doc(_SPACE_VOCAB, rng))
        labels.append(0)
    for _ in range(n_per_class):
        docs.append(_make_synthetic_doc(_HOCKEY_VOCAB, rng))
        labels.append(1)
    return docs, np.array(labels), "SYNTHETIC offline fallback corpus (not a real dataset)"


def load_text_classification_data(max_features=3000, test_size=0.25, random_state=42):
    """
    Returns:
        X_train, X_test: sparse TF-IDF matrices
        y_train, y_test: 0/1 labels
        vectorizer: fitted TfidfVectorizer
        source: string describing which data source was actually used
    """
    try:
        X_text, y, source = _load_real_20newsgroups()
        print(f"[data_utils] Loaded real dataset: {source}")
    except Exception as e:
        X_text, y, source = _load_synthetic()
        print(f"[data_utils] Network/dataset fetch failed ({e!r}); using {source}")

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, vectorizer, source


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vec, source = load_text_classification_data()
    print("Source:", source)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Feature dimensionality:", X_train.shape[1])
